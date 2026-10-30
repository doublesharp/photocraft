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

#[cfg(test)]
pub fn resolve(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>, platform: Platform, saved: Option<&Path>) -> Option<PathBuf> {
    resolve_with(env, exe, platform, saved, Path::is_file)
}

/// Apply a shared default only after explicit model/profile and portable locations.
pub fn resolve_with_shared(
    env: impl Fn(&str) -> Option<OsString>,
    exe: Option<&Path>,
    platform: Platform,
    saved: Option<&Path>,
    shared: Option<&Path>,
) -> Option<PathBuf> {
    resolve_shared_with(env, exe, platform, saved, shared, Path::is_file)
}

fn resolve_shared_with(
    env: impl Fn(&str) -> Option<OsString>,
    exe: Option<&Path>,
    platform: Platform,
    saved: Option<&Path>,
    shared: Option<&Path>,
    marker_exists: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let path = |key| env(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = path("PHOTOCRAFT_MODEL_DIR") {
        return Some(dir);
    }
    if let Some(dir) = saved.filter(|dir| dir.is_absolute()) {
        return Some(dir.to_path_buf());
    }
    if let Some(dir) = path("PHOTOCRAFT_CONFIG_DIR") {
        return Some(dir.join("models"));
    }
    if let Some(data) = portable_data_dir(exe, marker_exists) {
        // Keep portable copies self-contained even on read-only media. A shared default
        // must not redirect an explicit portable installation to the normal user profile.
        return Some(data.join("models"));
    }
    if let Some(shared) = shared.filter(|dir| dir.is_absolute()) {
        return Some(shared.to_path_buf());
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

/// An explicit AI directory wins; profiles and portable copies otherwise keep bindings beside
/// their own preferences. None means the normal suite-wide configuration location.
pub fn shared_store_path(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>) -> Option<PathBuf> {
    shared_store_path_with(env, exe, Path::is_file)
}

fn shared_store_path_with(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>, marker_exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let path = |key| env(key).filter(|value| !value.is_empty()).map(PathBuf::from);
    if let Some(dir) = path("CRAFT_AI_DIR") {
        return Some(dir.join("connections.json"));
    }
    if let Some(dir) = path("PHOTOCRAFT_CONFIG_DIR") {
        return Some(dir.join("ai/connections.json"));
    }
    portable_data_dir(exe, marker_exists).map(|dir| dir.join("ai/connections.json"))
}

fn portable_data_dir(exe: Option<&Path>, marker_exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    exe.and_then(Path::parent)
        .filter(|parent| ["portable.txt", "PhotoCraft.portable"].iter().any(|name| marker_exists(&parent.join(name))))
        .map(|parent| parent.join("PhotoCraftData"))
}

#[cfg(test)]
fn resolve_with(
    env: impl Fn(&str) -> Option<OsString>,
    exe: Option<&Path>,
    platform: Platform,
    saved: Option<&Path>,
    marker_exists: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    resolve_shared_with(env, exe, platform, saved, None, marker_exists)
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
        assert_eq!(resolve(env, None, Platform::Mac, None), Some(PathBuf::from("/chosen/models")));
    }
    #[test]
    fn config_override_keeps_models_inside_the_override() {
        assert_eq!(resolve(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/settings")]), None, Platform::Windows, None), Some(PathBuf::from("/settings/models")));
    }
    #[test]
    fn portable_models_follow_the_executable_unless_config_is_explicit() {
        let dir = Path::new("/portable");
        let executable = dir.join("photocraft");
        for marker in ["portable.txt", "PhotoCraft.portable"] {
            let exists = |path: &Path| path == dir.join(marker);
            assert_eq!(resolve_with(vars(&[("HOME", "/home/test")]), Some(&executable), Platform::Mac, None, exists), Some(dir.join("PhotoCraftData/models")));
            assert_eq!(
                resolve_with(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/custom")]), Some(&executable), Platform::Mac, None, exists),
                Some(PathBuf::from("/custom/models"))
            );
        }
    }
    #[test]
    fn windows_models_use_local_not_roaming_data() {
        assert_eq!(
            resolve(vars(&[("LOCALAPPDATA", "/local"), ("APPDATA", "/roaming")]), None, Platform::Windows, None),
            Some(PathBuf::from("/local/Photocraft/models"))
        );
    }
    #[test]
    fn xdg_data_and_home_fallback_are_non_roaming() {
        assert_eq!(
            resolve(vars(&[("XDG_DATA_HOME", "/data"), ("HOME", "/home/test")]), None, Platform::Xdg, None),
            Some(PathBuf::from("/data/photocraft/models"))
        );
        assert_eq!(
            resolve(vars(&[("XDG_DATA_HOME", "relative"), ("HOME", "/home/test")]), None, Platform::Xdg, None),
            Some(PathBuf::from("/home/test/.local/share/photocraft/models"))
        );
        assert_eq!(resolve(vars(&[]), None, Platform::Xdg, None), None);
    }
    #[test]
    fn mac_models_use_application_support() {
        assert_eq!(
            resolve(vars(&[("HOME", "/home/test")]), None, Platform::Mac, None),
            Some(PathBuf::from("/home/test/Library/Application Support/Photocraft/models"))
        );
    }
    #[test]
    fn saved_model_folder_is_reused_without_requiring_an_environment_override() {
        let saved = std::env::temp_dir().join("photocraft-existing-saved-models");
        for platform in [Platform::Mac, Platform::Windows, Platform::Xdg] {
            assert_eq!(resolve(vars(&[("HOME", "/home/test"), ("PHOTOCRAFT_CONFIG_DIR", "/settings")]), None, platform, Some(&saved)), Some(saved.clone()));
        }
        assert_eq!(resolve(vars(&[("PHOTOCRAFT_MODEL_DIR", "/explicit")]), None, Platform::Mac, Some(&saved)), Some(PathBuf::from("/explicit")));
        // An unavailable saved mount remains selected, rather than silently downloading elsewhere.
        assert_eq!(resolve(vars(&[]), None, Platform::Mac, Some(&saved)), Some(saved.clone()));
        assert_eq!(
            resolve(vars(&[("HOME", "/home/test")]), None, Platform::Mac, Some(Path::new("relative"))),
            Some(PathBuf::from("/home/test/Library/Application Support/Photocraft/models"))
        );
    }
    #[test]
    fn shared_default_cannot_override_explicit_profile_or_portable_models() {
        let shared = Path::new("/suite/models");
        let exe = Path::new("/portable/photocraft");
        let marker = |path: &Path| path == Path::new("/portable/portable.txt");
        assert_eq!(
            resolve_shared_with(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/profile")]), Some(exe), Platform::Mac, None, Some(shared), marker),
            Some(PathBuf::from("/profile/models"))
        );
        assert_eq!(
            resolve_shared_with(vars(&[("HOME", "/home/test")]), Some(exe), Platform::Mac, None, Some(shared), marker),
            Some(PathBuf::from("/portable/PhotoCraftData/models"))
        );
    }
    #[test]
    fn shared_default_preserves_explicit_model_and_saved_locations() {
        let shared = Path::new("/suite/models");
        assert_eq!(
            resolve_with_shared(vars(&[("PHOTOCRAFT_MODEL_DIR", "/explicit")]), None, Platform::Mac, Some(Path::new("/saved")), Some(shared)),
            Some(PathBuf::from("/explicit"))
        );
        assert_eq!(resolve_with_shared(vars(&[]), None, Platform::Mac, Some(Path::new("/saved")), Some(shared)), Some(PathBuf::from("/saved")));
    }
    #[test]
    fn ordinary_configuration_uses_the_shared_model_default() {
        for platform in [Platform::Mac, Platform::Windows, Platform::Xdg] {
            assert_eq!(
                resolve_with_shared(vars(&[("HOME", "/home/test"), ("LOCALAPPDATA", "/local")]), None, platform, None, Some(Path::new("/suite/models"))),
                Some(PathBuf::from("/suite/models"))
            );
        }
    }
    #[test]
    fn explicit_ai_directory_wins_over_profile_and_portable_store_paths() {
        let env = vars(&[("CRAFT_AI_DIR", "/ai"), ("PHOTOCRAFT_CONFIG_DIR", "/profile")]);
        let marker = |_: &Path| true;
        assert_eq!(shared_store_path_with(env, Some(Path::new("/portable/photocraft")), marker), Some(PathBuf::from("/ai/connections.json")));
    }
    #[test]
    fn profile_and_portable_stores_stay_beside_their_preferences() {
        let exe = Path::new("/portable/photocraft");
        for name in ["portable.txt", "PhotoCraft.portable"] {
            let marker = |path: &Path| path == Path::new("/portable").join(name);
            assert_eq!(
                shared_store_path_with(vars(&[("PHOTOCRAFT_CONFIG_DIR", "/profile")]), Some(exe), marker),
                Some(PathBuf::from("/profile/ai/connections.json"))
            );
            assert_eq!(shared_store_path_with(vars(&[]), Some(exe), marker), Some(PathBuf::from("/portable/PhotoCraftData/ai/connections.json")));
        }
        assert_eq!(shared_store_path(vars(&[("CRAFT_AI_DIR", ""), ("PHOTOCRAFT_CONFIG_DIR", "")]), None), None);
    }
}
