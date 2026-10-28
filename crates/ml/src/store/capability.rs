//! Store operations use directory handles; names never re-resolve through an ambient store path.
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::path::{Component, Path};

use cap_primitives::fs::{FollowSymlinks, open_dir_nofollow};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, DirBuilder, OpenOptions};
use same_file::Handle;

use crate::{Error, Result};

const MAX_DIRECTORY_DEPTH: usize = 256;

fn child_dir(parent: &Dir, name: &OsStr) -> io::Result<Dir> {
    let parent = parent.try_clone()?.into_std_file();
    open_dir_nofollow(&parent, Path::new(name)).map(Dir::from_std_file)
}

fn same_dir(a: &Dir, b: &Dir) -> io::Result<bool> {
    Ok(Handle::from_file(a.try_clone()?.into_std_file())? == Handle::from_file(b.try_clone()?.into_std_file())?)
}

pub(super) fn same_file(a: &File, b: &File) -> io::Result<bool> {
    Ok(Handle::from_file(a.try_clone()?)? == Handle::from_file(b.try_clone()?)?)
}

/// Keeps every parent alive as well as the directory itself. This permits detection of a
/// replaced ancestor while all actual I/O remains relative to the originally opened handles.
pub(super) struct StoreDir {
    parents: Vec<(Dir, OsString)>,
    pub(super) dir: Dir,
}

impl StoreDir {
    pub(super) fn acquire(path: &Path, create: bool) -> Result<Self> {
        Self::acquire_with(path, create, |_, _| {})
    }

    pub(super) fn acquire_with(path: &Path, create: bool, mut opened: impl FnMut(&Path, &Dir)) -> Result<Self> {
        if path.as_os_str().len() > 32_768 {
            return Err(Error::Storage("model path is too long".into()));
        }
        let absolute = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
        if absolute.as_os_str().len() > 32_768 {
            return Err(Error::Storage("model path is too long".into()));
        }
        let anchor = absolute.ancestors().last().ok_or_else(|| Error::Storage("invalid model directory".into()))?;
        let mut dir = Dir::open_ambient_dir(anchor, ambient_authority())?;
        let rest = absolute.strip_prefix(anchor).map_err(|_| Error::Storage("invalid model directory".into()))?;
        let mut parents = Vec::new();
        for component in rest.components() {
            let name = match component {
                Component::Normal(name) => name,
                Component::CurDir => continue,
                _ => return Err(Error::Storage("model directory must not contain parent or root components".into())),
            };
            if parents.len() >= MAX_DIRECTORY_DEPTH {
                return Err(Error::Storage("model directory is too deep".into()));
            }
            // Open existing ancestors first; creating an already-existing directory can
            // require unnecessary write permission on its parent on some platforms.
            let child = match child_dir(&dir, name) {
                Ok(child) => child,
                Err(e) if create && e.kind() == io::ErrorKind::NotFound => {
                    match dir.create_dir(name) {
                        Ok(()) => {}
                        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                        Err(e) => return Err(e.into()),
                    }
                    child_dir(&dir, name)?
                }
                Err(e) => return Err(e.into()),
            };
            // Neither the first nor the confirmation open follows a symlink. The second
            // detects replacement while the first directory capability was being acquired.
            opened(Path::new(name), &child);
            let confirmation = child_dir(&dir, name)?;
            if !same_dir(&child, &confirmation)? {
                return Err(Error::Storage("model directory changed while being opened".into()));
            }
            parents.push((dir, name.to_os_string()));
            dir = child;
        }
        let result = Self { parents, dir };
        result.ensure_attached()?;
        Ok(result)
    }

    pub(super) fn ensure_attached(&self) -> Result<()> {
        for (index, (parent, name)) in self.parents.iter().enumerate() {
            let expected = self.parents.get(index + 1).map(|(dir, _)| dir).unwrap_or(&self.dir);
            let current = child_dir(parent, name).map_err(|_| Error::Storage("model directory was replaced during the operation".into()))?;
            if !same_dir(&current, expected)? {
                return Err(Error::Storage("model directory was replaced during the operation".into()));
            }
        }
        Ok(())
    }

    pub(super) fn open_regular(&self, name: &OsStr) -> Result<File> {
        open_regular(&self.dir, name)
    }
}

fn open_regular(dir: &Dir, name: &OsStr) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true)._cap_fs_ext_follow(FollowSymlinks::No)._cap_fs_ext_nonblock(true);
    let file = dir.open_with(name, &options)?.into_std();
    if !file.metadata()?.is_file() {
        return Err(Error::Storage("model data must be a regular file, not a symbolic link or special entry".into()));
    }
    Ok(file)
}

fn random_name(prefix: &str, suffix: &str) -> Result<String> {
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|e| Error::Storage(format!("cannot create a temporary model name: {e}")))?;
    let mut name = String::from(prefix);
    for byte in random {
        use std::fmt::Write;
        write!(&mut name, "{byte:02x}").map_err(|_| Error::Storage("cannot create a temporary model name".into()))?;
    }
    name.push_str(suffix);
    Ok(name)
}

/// An exclusively created leaf. Cleanup stays on the held store handle and checks ownership
/// before unlinking; a concurrent writer can still race the name-based unlink itself.
pub(super) struct Partial<'a> {
    store: &'a StoreDir,
    pub(super) name: String,
    pub(super) file: File,
}

impl<'a> Partial<'a> {
    pub(super) fn create(store: &'a StoreDir) -> Result<Self> {
        for _ in 0..16 {
            let name = random_name(".photocraft-model-", ".partial")?;
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use cap_std::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match store.dir.open_with(&name, &options) {
                Ok(file) => return Ok(Self { store, name, file: file.into_std() }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err(Error::Storage("cannot create a unique temporary model file".into()))
    }

    pub(super) fn ensure_owned(&self) -> Result<()> {
        let current = self.store.open_regular(OsStr::new(&self.name))?;
        if !same_file(&current, &self.file)? {
            return Err(Error::Storage("temporary model file was replaced during installation".into()));
        }
        Ok(())
    }
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        if self.ensure_owned().is_ok() {
            let _ = self.store.dir.remove_file(&self.name);
            return;
        }
        // If a staged leaf was renamed, look for our open inode without following links.
        // Skip published model names and entries observed to be unrelated.
        if let Ok(entries) = self.store.dir.entries() {
            for entry in entries.take(4096).flatten() {
                let name = entry.file_name();
                if name == OsStr::new(&self.name) || Path::new(&name).extension() == Some(OsStr::new("onnx")) {
                    continue;
                }
                if let Ok(file) = self.store.open_regular(&name)
                    && same_file(&self.file, &file).unwrap_or(false)
                {
                    let _ = self.store.dir.remove_file(&name);
                }
            }
        }
    }
}

/// Removal first moves the named entry into an owned directory and checks the moved
/// object before deleting it. A raced replacement is restored without overwriting a new entry.
pub(super) struct Quarantine<'a> {
    store: &'a StoreDir,
    name: String,
    dir: Dir,
}

impl<'a> Quarantine<'a> {
    pub(super) fn create(store: &'a StoreDir) -> Result<Self> {
        for _ in 0..16 {
            let name = random_name(".photocraft-remove-", "")?;
            let builder = DirBuilder::new();
            #[cfg(unix)]
            let builder = {
                use cap_std::fs::DirBuilderExt;
                let mut builder = builder;
                builder.mode(0o700);
                builder
            };
            match store.dir.create_dir_with(&name, &builder) {
                Ok(()) => {
                    let dir = child_dir(&store.dir, OsStr::new(&name))?;
                    let stage = Self { store, name, dir };
                    stage.ensure_attached()?;
                    return Ok(stage);
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err(Error::Storage("cannot create a unique removal staging directory".into()))
    }

    pub(super) fn ensure_attached(&self) -> Result<()> {
        let current = child_dir(&self.store.dir, OsStr::new(&self.name)).map_err(|_| Error::Storage("removal staging directory was replaced".into()))?;
        if !same_dir(&current, &self.dir)? {
            return Err(Error::Storage("removal staging directory was replaced".into()));
        }
        Ok(())
    }

    pub(super) fn move_entry(&self, name: &str) -> Result<()> {
        self.store.dir.rename(name, &self.dir, "entry")?;
        Ok(())
    }

    pub(super) fn matches(&self, file: &File) -> Result<bool> {
        let moved = open_regular(&self.dir, OsStr::new("entry"))?;
        Ok(same_file(file, &moved)?)
    }

    pub(super) fn remove_entry(&self) -> Result<()> {
        self.dir.remove_file("entry")?;
        Ok(())
    }

    pub(super) fn restore(&self, name: &str) -> Result<()> {
        // No-clobber link works for regular files and symlinks without following them.
        self.dir.hard_link("entry", &self.store.dir, name)?;
        self.dir.remove_file("entry")?;
        Ok(())
    }

    pub(super) fn recovery_error(&self, error: Error) -> Error {
        Error::Storage(format!("{error}; the moved entry remains in model storage under {} for recovery", self.name))
    }
}

impl Drop for Quarantine<'_> {
    fn drop(&mut self) {
        if self.ensure_attached().is_ok() {
            // Only an empty directory is removed. Nonempty recovery data is retained.
            let _ = self.store.dir.remove_dir(&self.name);
        }
    }
}
