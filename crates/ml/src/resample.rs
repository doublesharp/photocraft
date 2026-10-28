//! Aspect-preserving model crops. No document-sized intermediate allocation is required here.
use crate::{Error, RemovalInput, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub side: usize,
    pub width: usize,
    pub height: usize,
    pub resized_width: usize,
    pub resized_height: usize,
    pub pad_x: usize,
    pub pad_y: usize,
    pub scale: f32,
}

impl Geometry {
    pub fn new(width: u32, height: u32, side: u32) -> Result<Self> {
        if width == 0 || height == 0 || side == 0 || side > 512 {
            return Err(Error::InvalidInput("invalid model crop dimensions".into()));
        }
        // Preserve small-crop detail; only large contexts are downsampled. Reflection padding
        // fills the remaining model field without distorting edges or stretching thin objects.
        let scale = (side as f64 / f64::from(width.max(height))).min(1.0);
        let resized_width = (f64::from(width) * scale).round().clamp(1., f64::from(side)) as usize;
        let resized_height = (f64::from(height) * scale).round().clamp(1., f64::from(side)) as usize;
        Ok(Self {
            side: side as usize,
            width: width as usize,
            height: height as usize,
            resized_width,
            resized_height,
            pad_x: (side as usize - resized_width) / 2,
            pad_y: (side as usize - resized_height) / 2,
            scale: scale as f32,
        })
    }

    pub fn prepare(&self, input: &RemovalInput) -> Result<(Vec<f32>, Vec<f32>)> {
        let pixels = self.side.checked_mul(self.side).ok_or_else(|| Error::InvalidInput("model crop is too large".into()))?;
        let mut rgb = vec![0.; pixels * 3];
        let mut mask = vec![0.; pixels];
        for y in 0..self.side {
            let ry = reflect(y as isize - self.pad_y as isize, self.resized_height);
            for x in 0..self.side {
                let rx = reflect(x as isize - self.pad_x as isize, self.resized_width);
                let source_x = ((rx as f64 + 0.5) * self.width as f64 / self.resized_width as f64 - 0.5).clamp(0., self.width.saturating_sub(1) as f64);
                let source_y = ((ry as f64 + 0.5) * self.height as f64 / self.resized_height as f64 - 0.5).clamp(0., self.height.saturating_sub(1) as f64);
                let i = y * self.side + x;
                for c in 0..3 {
                    let value = sample_interleaved(&input.rgb, self.width, self.height, source_x, source_y, c)?;
                    let dst = rgb.get_mut(c * pixels + i).ok_or_else(|| Error::InvalidInput("invalid RGB crop bounds".into()))?;
                    *dst = value;
                }
                // RGB padding is reflected, but its mask is known context. Reflecting the
                // mask would create duplicate holes and repeat work for narrow crops.
                if x < self.pad_x || x >= self.pad_x + self.resized_width || y < self.pad_y || y >= self.pad_y + self.resized_height {
                    continue;
                }
                // Conservative reduction keeps a narrow painted wire in the hole even when
                // downsampling. Known/protected pixels remain known in the original write mask.
                let x0 = rx * self.width / self.resized_width;
                let x1 = ((rx + 1) * self.width).div_ceil(self.resized_width).min(self.width);
                let y0 = ry * self.height / self.resized_height;
                let y1 = ((ry + 1) * self.height).div_ceil(self.resized_height).min(self.height);
                let mut hole = false;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        if input.mask.get(sy * self.width + sx).is_some_and(|v| *v > 0.) {
                            hole = true;
                            break;
                        }
                    }
                    if hole {
                        break;
                    }
                }
                if let Some(dst) = mask.get_mut(i) {
                    *dst = if hole { 1. } else { 0. };
                }
            }
        }
        if mask.iter().all(|v| *v > 0.) {
            return Err(Error::InvalidInput("the mask leaves no model context after crop reduction; use a smaller repair".into()));
        }
        Ok((rgb, mask))
    }

    pub fn restore(&self, planar: &[f32]) -> Result<Vec<f32>> {
        let pixels = self.side * self.side;
        if planar.len() != pixels * 3 {
            return Err(Error::Inference("unexpected generated image dimensions".into()));
        }
        let count =
            self.width.checked_mul(self.height).and_then(|v| v.checked_mul(3)).ok_or_else(|| Error::InvalidInput("output image is too large".into()))?;
        let mut out = vec![0.; count];
        for y in 0..self.height {
            for x in 0..self.width {
                let gx = (x as f64 + 0.5) * self.resized_width as f64 / self.width as f64 - 0.5 + self.pad_x as f64;
                let gy = (y as f64 + 0.5) * self.resized_height as f64 / self.height as f64 - 0.5 + self.pad_y as f64;
                for c in 0..3 {
                    let channel = planar.get(c * pixels..(c + 1) * pixels).ok_or_else(|| Error::Inference("invalid generated image channels".into()))?;
                    let value = sample_planar(channel, self.side, self.side, gx, gy)?;
                    let dst = out.get_mut((y * self.width + x) * 3 + c).ok_or_else(|| Error::Inference("invalid restored image bounds".into()))?;
                    *dst = value.clamp(0., 1.);
                }
            }
        }
        Ok(out)
    }
}

fn reflect(value: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1) as isize;
    let i = value.rem_euclid(period) as usize;
    if i < len { i } else { period as usize - i }
}

fn sample_interleaved(data: &[f32], w: usize, h: usize, x: f64, y: f64, c: usize) -> Result<f32> {
    bilinear(w, h, x, y, |px, py| data.get((py * w + px) * 3 + c).copied())
}
fn sample_planar(data: &[f32], w: usize, h: usize, x: f64, y: f64) -> Result<f32> {
    bilinear(w, h, x, y, |px, py| data.get(py * w + px).copied())
}
fn bilinear(w: usize, h: usize, x: f64, y: f64, at: impl Fn(usize, usize) -> Option<f32>) -> Result<f32> {
    let x = x.clamp(0., w.saturating_sub(1) as f64);
    let y = y.clamp(0., h.saturating_sub(1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;
    let a = at(x0, y0).ok_or_else(|| Error::InvalidInput("invalid crop buffer length".into()))?;
    let b = at(x1, y0).ok_or_else(|| Error::InvalidInput("invalid crop buffer length".into()))?;
    let c = at(x0, y1).ok_or_else(|| Error::InvalidInput("invalid crop buffer length".into()))?;
    let d = at(x1, y1).ok_or_else(|| Error::InvalidInput("invalid crop buffer length".into()))?;
    Ok((a + (b - a) * tx) * (1. - ty) + (c + (d - c) * tx) * ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_native_crop_preserves_pixels_and_aspect_ratio() {
        let input = RemovalInput { width: 3, height: 2, rgb: (0..18).map(|i| i as f32 / 18.).collect(), mask: vec![0., 1., 0., 0., 0., 0.] };
        let geometry = Geometry::new(3, 2, 512).unwrap();
        assert_eq!((geometry.resized_width, geometry.resized_height), (3, 2));
        let (rgb, mask) = geometry.prepare(&input).unwrap();
        assert!(mask.iter().any(|m| *m > 0.));
        assert_eq!(geometry.restore(&rgb).unwrap(), input.rgb);
    }
    #[test]
    fn reflected_image_padding_does_not_duplicate_the_hole() {
        let input = RemovalInput { width: 2, height: 1, rgb: vec![0.25; 6], mask: vec![1., 0.] };
        let geometry = Geometry::new(2, 1, 4).unwrap();
        let (_, mask) = geometry.prepare(&input).unwrap();
        assert_eq!(mask.iter().filter(|&&v| v > 0.).count(), 1, "padding must be known context, not repeated holes");
        assert_eq!(mask[geometry.pad_y * geometry.side + geometry.pad_x], 1.);
    }
    #[test]
    fn downsampling_does_not_erase_a_one_pixel_wire() {
        let input = RemovalInput { width: 1024, height: 2, rgb: vec![0.; 1024 * 2 * 3], mask: (0..2048).map(|i| if i == 3 { 1. } else { 0. }).collect() };
        let geometry = Geometry::new(1024, 2, 512).unwrap();
        let (_, mask) = geometry.prepare(&input).unwrap();
        assert_eq!((geometry.resized_width, geometry.resized_height), (512, 1));
        assert_eq!(geometry.scale, 0.5);
        assert!(mask.iter().any(|m| *m > 0.));
    }
}
