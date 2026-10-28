use super::*;
use crate::{NoopControl, RemovalService};
use std::io::Cursor;

// Admission is process-wide. Serialize tests that deliberately hold the shared permit.
static ADMISSION_TEST: Mutex<()> = Mutex::new(());

static FIXTURE: ModelSpec = ModelSpec {
    id: "fixture",
    label: "Fixture",
    description: "test data",
    bytes: 5,
    sha256: "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
    url: "https://example.invalid/pinned",
    license: "MIT",
    source: "test",
    backend: "test",
    input_size: 512,
    memory_mib: 1,
};

#[test]
fn incomplete_or_wrong_stream_never_publishes_model() {
    let temp = tempfile::tempdir().unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    for bytes in [b"mod".as_slice(), b"wrong", b"model-extra"] {
        assert!(service.publish(&FIXTURE, &mut Cursor::new(bytes), &NoopControl).is_err());
        assert!(!service.path(&FIXTURE, false).unwrap().exists());
        assert_eq!(fs::read_dir(temp.path().join("models")).unwrap().count(), 0, "partial files must be cleaned");
    }
}

#[test]
fn resource_permit_lasts_until_worker_return_even_when_cancelled() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let service = Arc::new(NativeService::new(None));
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (exit_tx, exit_rx) = std::sync::mpsc::channel();
    let worker = Arc::clone(&service);
    let thread = std::thread::spawn(move || {
        let lease = worker.begin().unwrap();
        entered_tx.send(()).unwrap();
        exit_rx.recv().unwrap();
        drop(lease);
    });
    entered_rx.recv().unwrap();
    assert!(matches!(service.begin(), Err(Error::Busy)));
    assert!(matches!(service.remove(&FIXTURE), Err(Error::Busy)));
    exit_tx.send(()).unwrap();
    thread.join().unwrap();
    assert!(service.begin().is_ok());
}

#[cfg(unix)]
#[test]
fn directory_and_model_symlinks_are_rejected_without_touching_target() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("original");
    fs::create_dir(&target).unwrap();
    let alias = temp.path().join("alias");
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    let service = NativeService::new(Some(alias));
    assert!(service.directory(true).is_err());
    let service = NativeService::new(Some(target.clone()));
    let victim = temp.path().join("victim");
    fs::write(&victim, b"model").unwrap();
    std::os::unix::fs::symlink(&victim, target.join(NativeService::file_name(&FIXTURE))).unwrap();
    assert!(service.remove(&FIXTURE).is_err());
    assert_eq!(fs::read(&victim).unwrap(), b"model");
}

#[test]
fn distinct_services_share_one_inference_and_management_permit() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let first = NativeService::new(None);
    let second = NativeService::new(None);
    let lease = first.begin().unwrap();
    assert!(matches!(second.begin(), Err(Error::Busy)), "a second service admitted overlapping model work");
    assert!(matches!(second.remove(&FIXTURE), Err(Error::Busy)), "model removal raced active work in another service");
    drop(lease);
    assert!(second.begin().is_ok());
}

#[test]
fn failed_reinstall_preserves_the_existing_verified_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    let path = service.path(&FIXTURE, false).unwrap();
    for bad in [b"wrong".as_slice(), b"mod".as_slice(), b"model-extra".as_slice()] {
        assert!(service.publish(&FIXTURE, &mut Cursor::new(bad), &NoopControl).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"model");
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }
}

#[test]
fn cancelled_noninterruptible_backend_keeps_global_permit_until_exit() {
    struct BlockingBackend {
        entered: std::sync::mpsc::Sender<()>,
        exit: Mutex<std::sync::mpsc::Receiver<()>>,
    }
    impl RemovalBackend for BlockingBackend {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(&self, _: &str, input: &RemovalInput, _: &dyn RemovalControl) -> Result<RemovalOutput> {
            self.entered.send(()).unwrap();
            self.exit.lock().unwrap().recv().unwrap();
            Ok(RemovalOutput { rgb: input.rgb.clone(), inference_scale: 1., load_ms: 0., forward_ms: 0. })
        }
    }
    struct Control(AtomicBool);
    impl RemovalControl for Control {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, _: &str) {}
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (exit_tx, exit_rx) = std::sync::mpsc::channel();
    let backend = Arc::new(BlockingBackend { entered: entered_tx, exit: Mutex::new(exit_rx) });
    let service = Arc::new(RemovalService::new(None).with_backend(backend));
    let control = Arc::new(Control(AtomicBool::new(false)));
    let worker_service = Arc::clone(&service);
    let worker_control = Arc::clone(&control);
    let worker = std::thread::spawn(move || {
        worker_service.infer("lama", &RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] }, worker_control.as_ref())
    });
    entered_rx.recv().unwrap();
    control.0.store(true, Ordering::Release);
    let other = RemovalService::new(None);
    assert!(matches!(other.remove("lama"), Err(Error::Busy)));
    assert!(matches!(other.install("lama", &NoopControl), Err(Error::Busy)));
    assert!(other.status().iter().all(|status| status.busy));
    exit_tx.send(()).unwrap();
    assert!(matches!(worker.join().unwrap(), Err(Error::Cancelled)));
    assert!(!other.status().iter().any(|status| status.busy));
}

#[cfg(unix)]
#[test]
fn directory_swap_during_install_cannot_publish_outside_the_store() {
    use std::os::unix::fs::symlink;
    struct Swap {
        root: PathBuf,
        moved: PathBuf,
        outside: PathBuf,
        done: AtomicBool,
    }
    impl RemovalControl for Swap {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, _: f32, _: &str) {
            if !self.done.swap(true, Ordering::AcqRel) {
                let partial = fs::read_dir(&self.root).unwrap().next().unwrap().unwrap().file_name();
                fs::rename(&self.root, &self.moved).unwrap();
                fs::write(self.outside.join(partial), b"wrong").unwrap();
                symlink(&self.outside, &self.root).unwrap();
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    let service = NativeService::new(Some(root.clone()));
    let control = Swap { root, moved: temp.path().join("moved"), outside: outside.clone(), done: AtomicBool::new(false) };
    let result = service.publish(&FIXTURE, &mut Cursor::new(b"model"), &control);
    assert!(!outside.join(NativeService::file_name(&FIXTURE)).exists(), "model publication escaped the managed directory");
    assert!(result.is_err(), "an installation whose directory was replaced must fail");
}

#[test]
fn malformed_adapter_outputs_and_diagnostics_are_rejected() {
    let input = RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] };
    for output in [
        RemovalOutput { rgb: vec![0.5; 5], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { rgb: vec![f32::NAN; 6], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { rgb: vec![1.1; 6], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { rgb: vec![0.5; 6], inference_scale: 1.1, load_ms: 0., forward_ms: 0. },
        RemovalOutput { rgb: vec![0.5; 6], inference_scale: 1., load_ms: f64::NAN, forward_ms: 0. },
        RemovalOutput { rgb: vec![0.5; 6], inference_scale: 1., load_ms: 0., forward_ms: -1. },
    ] {
        assert!(validate_output(&input, &output).is_err(), "invalid adapter output accepted: {output:?}");
    }
}

#[test]
fn changed_model_bytes_are_reverified_before_loading() {
    let temp = tempfile::tempdir().unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    service.verify_open(&FIXTURE, &NoopControl).unwrap();
    let path = service.path(&FIXTURE, false).unwrap();
    // Replacing the inode avoids reliance on filesystem timestamp granularity.
    let replacement = temp.path().join("replacement");
    fs::write(&replacement, b"wrong").unwrap();
    fs::rename(&replacement, &path).unwrap();
    assert!(matches!(service.verify_open(&FIXTURE, &NoopControl), Err(Error::Integrity(_))));
}

#[cfg(unix)]
#[test]
fn in_place_corruption_with_restored_mtime_is_not_trusted() {
    let temp = tempfile::tempdir().unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    let path = service.path(&FIXTURE, false).unwrap();
    service.verify_open(&FIXTURE, &NoopControl).unwrap();
    let original_mtime = fs::metadata(&path).unwrap().modified().unwrap();
    let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.write_all(b"wrong").unwrap();
    file.set_times(fs::FileTimes::new().set_modified(original_mtime)).unwrap();
    assert!(matches!(service.verify_open(&FIXTURE, &NoopControl), Err(Error::Integrity(_))));
    assert!(!service.state().verified.contains_key(FIXTURE.id), "a failed verification must clear cached trust");
}

#[test]
fn model_parse_snapshot_is_the_exact_bytes_that_were_hashed() {
    struct Rewrite {
        path: PathBuf,
        done: AtomicBool,
    }
    impl RemovalControl for Rewrite {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, _: f32, _: &str) {
            if !self.done.swap(true, Ordering::AcqRel) {
                fs::write(&self.path, b"wrong").unwrap();
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model");
    fs::write(&path, b"model").unwrap();
    let mut file = File::open(&path).unwrap();
    let ctl = Rewrite { path: path.clone(), done: AtomicBool::new(false) };
    let bytes = verified_snapshot(&mut file, &FIXTURE, &ctl).unwrap();
    assert_eq!(bytes, b"model", "parsing must use the hashed snapshot even if the file changes");
    assert_eq!(fs::read(path).unwrap(), b"wrong");
}

#[test]
fn early_admission_bounds_memory_and_holds_process_wide_permit() {
    struct Echo;
    impl RemovalBackend for Echo {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(&self, _: &str, input: &RemovalInput, _: &dyn RemovalControl) -> Result<RemovalOutput> {
            Ok(RemovalOutput { rgb: input.rgb.clone(), inference_scale: 1., load_ms: 0., forward_ms: 0. })
        }
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let service = RemovalService::new(None).with_backend(Arc::new(Echo));
    let other = RemovalService::new(None).with_backend(Arc::new(Echo));
    let model = crate::catalog::find("lama").unwrap();
    let working = u64::from(model.memory_mib) * 1024 * 1024;
    let input = RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] };
    assert!(service.admit("lama", 56, working + 55).is_err());
    assert!(service.admit("lama", u64::MAX, u64::MAX).is_err());
    let mut too_small = service.admit("lama", 55, working + 55).unwrap();
    assert!(too_small.infer(&input, &NoopControl).is_err());
    drop(too_small);
    let mut permit = service.admit("lama", 56, working + 56).unwrap();
    assert!(matches!(other.admit("lama", 56, u64::MAX), Err(Error::Busy)));
    assert!(matches!(other.remove("lama"), Err(Error::Busy)));
    assert_eq!(permit.infer(&input, &NoopControl).unwrap().rgb, input.rgb);
    assert!(permit.infer(&input, &NoopControl).is_err(), "a permit must not accumulate multiple outputs");
    assert!(matches!(other.admit("lama", 56, u64::MAX), Err(Error::Busy)));
    assert!(matches!(service.admit("unknown", 0, u64::MAX), Err(Error::UnknownModel(_))));
    let reservation = permit.into_reservation();
    drop(service);
    assert!(matches!(other.admit("lama", 56, u64::MAX), Err(Error::Busy)), "the output reservation must outlive its service");
    drop(reservation);
    assert!(other.admit("lama", 56, u64::MAX).is_ok());
}

#[cfg(unix)]
#[test]
fn acquisition_rejects_directory_replaced_before_confirmation() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let moved = temp.path().join("moved");
    let outside = temp.path().join("outside");
    fs::create_dir(&root).unwrap();
    fs::create_dir(&outside).unwrap();
    let result = StoreDir::acquire_with(&root, false, |name, _| {
        if name == Path::new("models") {
            fs::rename(&root, &moved).unwrap();
            symlink(&outside, &root).unwrap();
        }
    });
    assert!(result.is_err());
    assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn staged_leaf_replacement_is_rejected_and_owned_partial_is_cleaned() {
    use std::os::unix::fs::symlink;
    struct Swap {
        root: PathBuf,
        victim: PathBuf,
        done: AtomicBool,
    }
    impl RemovalControl for Swap {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, _: f32, _: &str) {
            if !self.done.swap(true, Ordering::AcqRel) {
                let partial = fs::read_dir(&self.root).unwrap().next().unwrap().unwrap().path();
                fs::rename(&partial, self.root.join("moved-partial")).unwrap();
                symlink(&self.victim, partial).unwrap();
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let victim = temp.path().join("victim");
    fs::write(&victim, b"wrong").unwrap();
    let ctl = Swap { root: root.clone(), victim: victim.clone(), done: AtomicBool::new(false) };
    let service = NativeService::new(Some(root.clone()));
    assert!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &ctl).is_err());
    assert!(!root.join(NativeService::file_name(&FIXTURE)).exists());
    assert!(!root.join("moved-partial").exists());
    assert_eq!(fs::read(victim).unwrap(), b"wrong");
}

#[cfg(unix)]
#[test]
fn model_open_rejects_leaf_and_directory_swaps_during_verification() {
    use std::os::unix::fs::symlink;
    struct Swap {
        root: PathBuf,
        moved: PathBuf,
        outside: PathBuf,
        leaf: bool,
        done: AtomicBool,
    }
    impl RemovalControl for Swap {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, _: f32, _: &str) {
            if !self.done.swap(true, Ordering::AcqRel) {
                if self.leaf {
                    let model = self.root.join(NativeService::file_name(&FIXTURE));
                    fs::rename(&model, &self.moved).unwrap();
                    symlink(&self.outside, &model).unwrap();
                } else {
                    fs::rename(&self.root, &self.moved).unwrap();
                    symlink(&self.outside, &self.root).unwrap();
                }
            }
        }
    }
    for leaf in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let outside = temp.path().join("outside");
        if leaf {
            fs::write(&outside, b"wrong").unwrap();
        } else {
            fs::create_dir(&outside).unwrap();
        }
        let service = NativeService::new(Some(root.clone()));
        service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
        let ctl = Swap { root, moved: temp.path().join("moved"), outside: outside.clone(), leaf, done: AtomicBool::new(false) };
        assert!(service.verify_open(&FIXTURE, &ctl).is_err());
        assert!(!service.state().verified.contains_key(FIXTURE.id));
        if leaf {
            assert_eq!(fs::read(outside).unwrap(), b"wrong");
        } else {
            assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
        }
    }
}

#[cfg(unix)]
#[test]
fn remove_rejects_leaf_and_directory_swaps_before_unlink() {
    use std::os::unix::fs::symlink;
    let _serial = ADMISSION_TEST.lock().unwrap();
    for leaf in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let outside = temp.path().join("outside");
        if leaf {
            fs::write(&outside, b"victim").unwrap();
        } else {
            fs::create_dir(&outside).unwrap();
        }
        let service = NativeService::new(Some(root.clone()));
        service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
        let dir = service.directory(false).unwrap();
        let moved = temp.path().join("moved");
        let model = root.join(NativeService::file_name(&FIXTURE));
        let result = service.remove_in(&dir, &FIXTURE, || {
            if leaf {
                fs::rename(&model, &moved).unwrap();
                symlink(&outside, &model).unwrap();
            } else {
                fs::rename(&root, &moved).unwrap();
                symlink(&outside, &root).unwrap();
            }
        });
        assert!(result.is_err());
        if leaf {
            assert!(fs::symlink_metadata(&model).unwrap().file_type().is_symlink());
            assert_eq!(fs::read(&outside).unwrap(), b"victim");
            assert_eq!(fs::read(&moved).unwrap(), b"model");
        } else {
            assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
            assert_eq!(fs::read(moved.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
        }
    }
}

#[test]
fn failed_stream_and_cancellation_leave_an_existing_install_untouched() {
    struct Fail;
    impl Read for Fail {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk failed"))
        }
    }
    struct Cancel(AtomicBool);
    impl RemovalControl for Cancel {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, _: &str) {
            self.0.store(true, Ordering::Release);
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    assert!(service.publish(&FIXTURE, &mut Fail, &NoopControl).is_err());
    assert!(matches!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &Cancel(AtomicBool::new(false))), Err(Error::Cancelled)));
    assert_eq!(fs::read(root.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
    assert_eq!(fs::read_dir(root).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn directory_swap_after_publication_is_reported_and_rolls_back_own_file() {
    use std::os::unix::fs::symlink;
    struct Swap {
        root: PathBuf,
        moved: PathBuf,
        outside: PathBuf,
    }
    impl RemovalControl for Swap {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, _: f32, label: &str) {
            if label == "Model installed" {
                fs::rename(&self.root, &self.moved).unwrap();
                symlink(&self.outside, &self.root).unwrap();
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    let ctl = Swap { root: root.clone(), moved: temp.path().join("moved"), outside: outside.clone() };
    let service = NativeService::new(Some(root));
    assert!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &ctl).is_err());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    assert_eq!(fs::read_dir(ctl.moved).unwrap().count(), 0);
}

#[test]
fn oversized_model_identifier_and_storage_path_fail_without_copying_input() {
    let id = "x".repeat(129);
    assert!(matches!(crate::catalog::find(&id), Err(Error::InvalidInput(_))));
    let name = "x".repeat(32_769);
    assert!(StoreDir::acquire(Path::new(&name), true).is_err());
}

#[cfg(unix)]
#[test]
fn staged_leaf_swapped_after_last_check_is_not_left_published() {
    use std::os::unix::fs::symlink;
    for link in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let victim = temp.path().join("victim");
        fs::write(&victim, b"wrong").unwrap();
        let service = NativeService::new(Some(root.clone()));
        let dir = service.directory(true).unwrap();
        let result = service.publish_in_with(&dir, &FIXTURE, &mut Cursor::new(b"model"), &NoopControl, || {
            let partial = fs::read_dir(&root).unwrap().next().unwrap().unwrap().path();
            fs::rename(&partial, root.join("moved-partial")).unwrap();
            if link {
                symlink(&victim, partial).unwrap();
            } else {
                fs::write(partial, b"wrong").unwrap();
            }
        });
        assert!(result.is_err());
        assert!(!root.join(NativeService::file_name(&FIXTURE)).exists());
        assert!(fs::symlink_metadata(root.join(NativeService::file_name(&FIXTURE))).is_err());
        assert!(!root.join("moved-partial").exists());
        assert_eq!(fs::read(victim).unwrap(), b"wrong");
    }
}

#[test]
fn cancellation_after_publication_does_not_reverse_commit() {
    struct CancelAtEnd(AtomicBool);
    impl RemovalControl for CancelAtEnd {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, label: &str) {
            if label == "Model installed" {
                self.0.store(true, Ordering::Release);
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    assert!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &CancelAtEnd(AtomicBool::new(false))).is_ok());
    assert_eq!(fs::read(root.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
}

#[cfg(unix)]
#[test]
fn failed_directory_acquisition_clears_previous_verified_status() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    let service = NativeService::new(Some(root.clone()));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    service.verify_open(&FIXTURE, &NoopControl).unwrap();
    fs::rename(&root, temp.path().join("moved")).unwrap();
    symlink(&outside, &root).unwrap();
    assert!(service.verify_open(&FIXTURE, &NoopControl).is_err());
    assert!(!service.state().verified.contains_key(FIXTURE.id));
}

#[cfg(unix)]
#[test]
fn install_uses_held_directory_even_when_downloader_swaps_root() {
    use std::os::unix::fs::symlink;
    struct Downloader {
        root: PathBuf,
        moved: PathBuf,
        outside: PathBuf,
    }
    impl ModelDownloader for Downloader {
        fn open(&self, _: &ModelSpec) -> Result<Box<dyn Read + Send>> {
            fs::rename(&self.root, &self.moved)?;
            symlink(&self.outside, &self.root)?;
            Ok(Box::new(Cursor::new(b"model")))
        }
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    let service = NativeService::new(Some(root.clone()));
    service.set_downloader(Arc::new(Downloader { root, moved: temp.path().join("moved"), outside: outside.clone() }));
    assert!(service.install(&FIXTURE, &NoopControl).is_err());
    assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
    assert_eq!(fs::read_dir(temp.path().join("moved")).unwrap().count(), 0);
}

#[test]
fn cancellation_immediately_before_publication_never_publishes() {
    struct CancelBeforeLink {
        cancelled: AtomicBool,
        published: AtomicBool,
    }
    impl RemovalControl for CancelBeforeLink {
        fn cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, label: &str) {
            if label == "Model installed" {
                self.published.store(true, Ordering::Release);
            }
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    let dir = service.directory(true).unwrap();
    let ctl = CancelBeforeLink { cancelled: AtomicBool::new(false), published: AtomicBool::new(false) };
    let result = service.publish_in_with(&dir, &FIXTURE, &mut Cursor::new(b"model"), &ctl, || {
        ctl.cancelled.store(true, Ordering::Release);
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert!(!ctl.published.load(Ordering::Acquire), "publication happened after cancellation");
    assert_eq!(fs::read_dir(root).unwrap().count(), 0);
}

#[test]
fn cancellation_after_irreversible_gate_cannot_turn_install_into_cancelled() {
    struct Gate {
        cancelled: AtomicBool,
        entered: AtomicBool,
    }
    impl RemovalControl for Gate {
        fn cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, _: &str) {}
        fn begin_irreversible(&self) -> Result<()> {
            self.entered.store(true, Ordering::Release);
            // Simulates a late cancellation signal after the host has closed its gate.
            self.cancelled.store(true, Ordering::Release);
            Ok(())
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    let ctl = Gate { cancelled: AtomicBool::new(false), entered: AtomicBool::new(false) };
    assert!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &ctl).is_ok());
    assert!(ctl.entered.load(Ordering::Acquire));
    assert!(ctl.cancelled.load(Ordering::Acquire));
    assert_eq!(fs::read(root.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
}

#[test]
fn remove_success_deletes_only_the_model_and_its_staging_directory() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    fs::write(root.join("other"), b"keep").unwrap();
    service.remove(&FIXTURE).unwrap();
    assert_eq!(fs::read(root.join("other")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(root).unwrap().count(), 1);
}

#[test]
fn concurrent_cancellation_is_refused_after_publication_gate() {
    struct Gate {
        state: Mutex<(bool, bool)>, // cancelled, committing
        entered: std::sync::mpsc::Sender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }
    impl Gate {
        fn try_cancel(&self) -> bool {
            let mut state = self.state.lock().unwrap();
            if state.1 {
                false
            } else {
                state.0 = true;
                true
            }
        }
    }
    impl RemovalControl for Gate {
        fn cancelled(&self) -> bool {
            self.state.lock().unwrap().0
        }
        fn progress(&self, _: f32, _: &str) {}
        fn begin_irreversible(&self) -> Result<()> {
            let mut state = self.state.lock().unwrap();
            if state.0 {
                return Err(Error::Cancelled);
            }
            state.1 = true;
            drop(state);
            self.entered.send(()).unwrap();
            self.release.lock().unwrap().recv().unwrap();
            Ok(())
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let control = Arc::new(Gate { state: Mutex::new((false, false)), entered: entered_tx, release: Mutex::new(release_rx) });
    let worker_control = Arc::clone(&control);
    let worker_root = root.clone();
    let worker = std::thread::spawn(move || NativeService::new(Some(worker_root)).publish(&FIXTURE, &mut Cursor::new(b"model"), worker_control.as_ref()));
    entered_rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
    assert!(!control.try_cancel());
    release_tx.send(()).unwrap();
    assert!(worker.join().unwrap().is_ok());
    assert_eq!(fs::read(root.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
}

#[test]
fn removal_recovery_never_overwrites_a_concurrent_new_entry() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    let dir = service.directory(false).unwrap();
    let name = NativeService::file_name(&FIXTURE);
    let quarantine = Quarantine::create(&dir).unwrap();
    quarantine.move_entry(&name).unwrap();
    fs::write(root.join(&name), b"newer").unwrap();
    assert!(quarantine.restore(&name).is_err());
    drop(quarantine);
    assert_eq!(fs::read(root.join(name)).unwrap(), b"newer");
    let recovery = fs::read_dir(&root).unwrap().map(|entry| entry.unwrap().path()).find(|path| path.is_dir()).unwrap();
    assert_eq!(fs::read(recovery.join("entry")).unwrap(), b"model");
}

#[test]
fn late_cancellation_does_not_fail_the_idempotent_publication_path() {
    struct Gate(AtomicBool);
    impl RemovalControl for Gate {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, _: &str) {}
        fn begin_irreversible(&self) -> Result<()> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = NativeService::new(Some(root.clone()));
    service.publish(&FIXTURE, &mut Cursor::new(b"model"), &NoopControl).unwrap();
    assert!(service.publish(&FIXTURE, &mut Cursor::new(b"model"), &Gate(AtomicBool::new(false))).is_ok());
    assert_eq!(fs::read(root.join(NativeService::file_name(&FIXTURE))).unwrap(), b"model");
    assert_eq!(fs::read_dir(root).unwrap().count(), 1);
}

#[test]
fn review_truncated_catalog_file_remains_visible_and_removable() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    let model = &crate::catalog::CATALOG[0];
    let path = service.path(model, true).unwrap();
    fs::write(&path, b"short").unwrap();
    let status = service.status(true).into_iter().find(|status| status.id == model.id).unwrap();
    assert!(status.installed, "the manager must be able to remove a corrupt owned artifact");
    assert!(!status.available && !status.verified);
    assert!(status.reason.unwrap().contains("size"));
    service.remove(model).unwrap();
    assert!(!path.exists());
}
