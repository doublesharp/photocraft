//! Model storage stays outside roaming configuration; paths are resolved without creating files.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub enum Platform {
    Mac,
    Windows,
    Xdg,
}

impl Platform {
    pub fn current() -> Self {
        match std::env::consts::OS {
            "macos" => Self::Mac,
            "windows" => Self::Windows,
            _ => Self::Xdg,
        }
    }
}

pub fn resolve(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>, platform: Platform) -> Option<PathBuf> {
    resolve_with(env, exe, platform, Path::is_file)
}

fn resolve_with(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>, platform: Platform, marker_exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let path = |key| env(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = path("PHOTOCRAFT_MODEL_DIR") {
        return Some(dir);
    }
    if let Some(dir) = path("PHOTOCRAFT_CONFIG_DIR") {
        return Some(dir.join("models"));
    }
    if let Some(parent) = exe.and_then(Path::parent)
        && ["portable.txt", "PhotoCraft.portable"].iter().any(|name| marker_exists(&parent.join(name)))
    {
        // Retain this path even on read-only media: only an explicit user override may
        // redirect model installation away from the portable application's directory.
        return Some(parent.join("PhotoCraftData").join("models"));
    }
    match platform {
        Platform::Mac => path("HOME").map(|home| home.join("Library/Application Support/Photocraft/models")),
        Platform::Windows => path("LOCALAPPDATA").map(|local| local.join("Photocraft/models")),
        Platform::Xdg => path("XDG_DATA_HOME")
            .filter(|p| p.is_absolute())
            .or_else(|| path("HOME").map(|home| home.join(".local/share")))
            .map(|data| data.join("photocraft/models")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vars<'a>(pairs: &'a [(&str, &str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |k| pairs.iter().find(|(key, _)| *key == k).map(|(_, value)| OsString::from(value))
    }
    #[test]
    fn explicit_model_location_wins_over_config_and_platform() {
        let env = vars(&[("PHOTOCRAFT_MODEL_DIR", "/chosen/models"), ("PHOTOCRAFT_CONFIG_DIR", "/chosen/config"), ("HOME", "/home/test")]);
        assert_eq!(resolve(env, None, Platform::Mac), Some(PathBuf::from("/chosen/models")));
    }
    #[test]
    fn config_override_keeps_models_inside_the_override() {
        assert_eq!(resolve(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/settings")]), None, Platform::Windows), Some(PathBuf::from("/settings/models")));
    }
    #[test]
    fn portable_models_follow_the_executable_unless_config_is_explicit() {
        let dir = Path::new("/portable");
        let executable = dir.join("photocraft");
        for marker in ["portable.txt", "PhotoCraft.portable"] {
            let exists = |path: &Path| path == dir.join(marker);
            assert_eq!(resolve_with(vars(&[("HOME", "/home/test")]), Some(&executable), Platform::Mac, exists), Some(dir.join("PhotoCraftData/models")));
            assert_eq!(
                resolve_with(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/custom")]), Some(&executable), Platform::Mac, exists),
                Some(PathBuf::from("/custom/models"))
            );
        }
    }
    #[test]
    fn windows_models_use_local_not_roaming_data() {
        assert_eq!(
            resolve(vars(&[("LOCALAPPDATA", "/local"), ("APPDATA", "/roaming")]), None, Platform::Windows),
            Some(PathBuf::from("/local/Photocraft/models"))
        );
    }
    #[test]
    fn xdg_data_and_home_fallback_are_non_roaming() {
        assert_eq!(resolve(vars(&[("XDG_DATA_HOME", "/data"), ("HOME", "/home/test")]), None, Platform::Xdg), Some(PathBuf::from("/data/photocraft/models")));
        assert_eq!(
            resolve(vars(&[("XDG_DATA_HOME", "relative"), ("HOME", "/home/test")]), None, Platform::Xdg),
            Some(PathBuf::from("/home/test/.local/share/photocraft/models"))
        );
        assert_eq!(resolve(vars(&[]), None, Platform::Xdg), None);
    }
    #[test]
    fn mac_models_use_application_support() {
        assert_eq!(
            resolve(vars(&[("HOME", "/home/test")]), None, Platform::Mac),
            Some(PathBuf::from("/home/test/Library/Application Support/Photocraft/models"))
        );
    }
}
