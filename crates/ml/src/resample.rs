//! Aspect-preserving model crops. No document-sized intermediate allocation is required here.
use crate::{Error, RemovalInput, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub side: usize,
    pub rows: usize,
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
            rows: side as usize,
            width: width as usize,
            height: height as usize,
            resized_width,
            resized_height,
            pad_x: (side as usize - resized_width) / 2,
            pad_y: (side as usize - resized_height) / 2,
            scale: scale as f32,
        })
    }

    #[cfg(any(test, feature = "osor"))]
    pub fn osor(width: u32, height: u32) -> Result<Self> {
        let mut geometry = Self::new(width, height, 512)?;
        geometry.side = geometry.resized_width.div_ceil(8).saturating_mul(8).max(32);
        geometry.rows = geometry.resized_height.div_ceil(8).saturating_mul(8).max(32);
        geometry.pad_x = (geometry.side - geometry.resized_width) / 2;
        geometry.pad_y = (geometry.rows - geometry.resized_height) / 2;
        Ok(geometry)
    }

    pub fn prepare(&self, input: &RemovalInput) -> Result<(Vec<f32>, Vec<f32>)> {
        let pixels = self.side.checked_mul(self.rows).ok_or_else(|| Error::InvalidInput("model crop is too large".into()))?;
        let mut rgb = vec![0.; pixels * 3];
        let mut mask = vec![0.; pixels];
        self.prepare_mask(&input.mask, &mut mask)?;
        for y in 0..self.rows {
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
                // A reflected sample has the same validity as the source it came from.
                // Treating reflected object pixels as known context can make a model copy
                // the object back into the repair. Read only the precomputed interior mask.
                let source = (self.pad_y + ry) * self.side + self.pad_x + rx;
                let coverage = mask.get(source).copied().ok_or_else(|| Error::InvalidInput("invalid reflected mask bounds".into()))?;
                let dst = mask.get_mut(i).ok_or_else(|| Error::InvalidInput("invalid padded mask bounds".into()))?;
                *dst = coverage;
            }
        }
        if mask.iter().all(|v| *v > 0.) {
            return Err(Error::InvalidInput("the mask leaves no model context after crop reduction; use a smaller repair".into()));
        }
        Ok((rgb, mask))
    }

    fn prepare_mask(&self, source: &[f32], mask: &mut [f32]) -> Result<()> {
        // Reduce each real source footprint once. Reflecting this reduced mask alongside RGB
        // keeps preprocessing bounded even for a long, thin crop with extensive padding.
        for ry in 0..self.resized_height {
            for rx in 0..self.resized_width {
                // Conservative reduction keeps a one-pixel painted wire in the hole.
                let x0 = rx * self.width / self.resized_width;
                let x1 = ((rx + 1) * self.width).div_ceil(self.resized_width).min(self.width);
                let y0 = ry * self.height / self.resized_height;
                let y1 = ((ry + 1) * self.height).div_ceil(self.resized_height).min(self.height);
                let mut hole = false;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        let coverage = source.get(sy * self.width + sx).ok_or_else(|| Error::InvalidInput("invalid source mask length".into()))?;
                        if *coverage > 0. {
                            hole = true;
                            break;
                        }
                    }
                    if hole {
                        break;
                    }
                }
                let index = (self.pad_y + ry) * self.side + self.pad_x + rx;
                let dst = mask.get_mut(index).ok_or_else(|| Error::InvalidInput("invalid reduced mask bounds".into()))?;
                *dst = if hole { 1. } else { 0. };
            }
        }
        Ok(())
    }

    #[cfg(any(test, feature = "osor"))]
    pub fn restore_alpha(&self, planar: &[f32]) -> Result<Vec<f32>> {
        self.restore_channels(planar, 1)
    }

    pub fn restore(&self, planar: &[f32]) -> Result<Vec<f32>> {
        self.restore_channels(planar, 3)
    }

    fn restore_channels(&self, planar: &[f32], channels: usize) -> Result<Vec<f32>> {
        let pixels = self.side * self.rows;
        if planar.len() != pixels * channels {
            return Err(Error::Inference("unexpected generated image dimensions".into()));
        }
        let count =
            self.width.checked_mul(self.height).and_then(|v| v.checked_mul(channels)).ok_or_else(|| Error::InvalidInput("output image is too large".into()))?;
        let mut out = vec![0.; count];
        for y in 0..self.height {
            for x in 0..self.width {
                let gx = (x as f64 + 0.5) * self.resized_width as f64 / self.width as f64 - 0.5 + self.pad_x as f64;
                let gy = (y as f64 + 0.5) * self.resized_height as f64 / self.height as f64 - 0.5 + self.pad_y as f64;
                for c in 0..channels {
                    let channel = planar.get(c * pixels..(c + 1) * pixels).ok_or_else(|| Error::Inference("invalid generated image channels".into()))?;
                    let value = sample_planar(channel, self.side, self.rows, gx, gy)?;
                    let dst = out.get_mut((y * self.width + x) * channels + c).ok_or_else(|| Error::Inference("invalid restored image bounds".into()))?;
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
    fn osor_crop_pads_axes_to_eight_without_upscaling_small_inputs() {
        let geometry = Geometry::osor(101, 47).unwrap();
        assert_eq!((geometry.side, geometry.rows), (104, 48));
        assert_eq!((geometry.resized_width, geometry.resized_height), (101, 47));
        let thin = Geometry::osor(2000, 1).unwrap();
        assert_eq!((thin.side, thin.rows), (512, 32));
        assert_eq!((thin.resized_width, thin.resized_height), (512, 1));
        let input = RemovalInput { width: 3, height: 2, rgb: (0..18).map(|i| i as f32 / 18.).collect(), mask: vec![0., 0.25, 1., 0., 0., 0.] };
        let small = Geometry::osor(3, 2).unwrap();
        let (rgb, _) = small.prepare(&input).unwrap();
        assert_eq!(small.restore(&rgb).unwrap(), input.rgb);
        let alpha = vec![0.25; small.side * small.rows];
        assert_eq!(small.restore_alpha(&alpha).unwrap(), vec![0.25; 6]);
        assert!(small.restore_alpha(&alpha[..alpha.len() - 1]).is_err());
    }

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
    fn reflected_image_padding_preserves_unknown_coverage() {
        let input = RemovalInput { width: 2, height: 1, rgb: vec![0.25; 6], mask: vec![1., 0.] };
        let geometry = Geometry::new(2, 1, 4).unwrap();
        let (_, mask) = geometry.prepare(&input).unwrap();
        assert_eq!(mask.iter().filter(|&&v| v > 0.).count(), 8, "reflected unknown pixels stay masked");
        assert_eq!(mask[geometry.pad_y * geometry.side + geometry.pad_x], 1.);
    }
    #[test]
    fn reflected_padding_does_not_expose_masked_source() {
        // A small crop is reflected extensively for LaMa and rounded to a multiple of
        // eight for OSOR. Neither may present reflected object pixels as known context.
        for geometry in [Geometry::new(3, 2, 8).unwrap(), Geometry::osor(3, 2).unwrap()] {
            let mut input = RemovalInput { width: 3, height: 2, rgb: vec![0.25; 18], mask: vec![0., 1., 0., 0., 0., 0.] };
            let (before, mask) = geometry.prepare(&input).unwrap();
            input.rgb[3..6].fill(1.);
            let (after, after_mask) = geometry.prepare(&input).unwrap();
            assert_eq!(mask, after_mask);
            let pixels = geometry.side * geometry.rows;
            for (i, coverage) in mask.iter().enumerate() {
                if *coverage == 0. {
                    for channel in 0..3 {
                        assert_eq!(before[channel * pixels + i], after[channel * pixels + i], "masked source leaked into known model context at {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn reflected_padding_cannot_invent_context_lost_during_reduction() {
        let mut input = RemovalInput { width: 1024, height: 1, rgb: vec![0.25; 3072], mask: vec![1.; 1024] };
        input.mask[0] = 0.;
        input.validate().unwrap();
        // The only known source pixel shares a reduced footprint with an unknown one.
        // Padding cannot rescue the resulting context-free model input.
        for geometry in [Geometry::new(1024, 1, 512).unwrap(), Geometry::osor(1024, 1).unwrap()] {
            assert!(matches!(geometry.prepare(&input), Err(Error::InvalidInput(_))));
        }
    }

    #[test]
    fn downsampled_and_padded_known_pixels_exclude_masked_rgb() {
        for (width, height, side) in [(17, 5, 8), (5, 17, 8), (1025, 3, 512), (1, 1025, 512)] {
            let count = (width * height) as usize;
            let mut input = RemovalInput { width, height, rgb: vec![0.; count * 3], mask: vec![0.; count] };
            // Include image edges and fractional coverage: every positive source coverage
            // is unknown to the model, while the original write mask stays unchanged.
            for i in [0, count / 2, count - 1] {
                input.mask[i] = 0.25;
                input.rgb[i * 3..(i + 1) * 3].fill(1.);
            }
            let geometry = Geometry::new(width, height, side).unwrap();
            let (rgb, mask) = geometry.prepare(&input).unwrap();
            let pixels = geometry.side * geometry.rows;
            assert!(mask.contains(&0.));
            for (i, coverage) in mask.iter().enumerate() {
                if *coverage == 0. {
                    for channel in 0..3 {
                        assert_eq!(rgb[channel * pixels + i], 0., "unknown RGB leaked at {width}x{height}, pixel {i}");
                    }
                }
            }
        }
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
