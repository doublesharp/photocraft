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
            Ok(RemovalOutput { edit_alpha: None, rgb: input.rgb.clone(), inference_scale: 1., load_ms: 0., forward_ms: 0. })
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
        RemovalOutput { edit_alpha: None, rgb: vec![0.5; 5], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { edit_alpha: None, rgb: vec![f32::NAN; 6], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { edit_alpha: None, rgb: vec![1.1; 6], inference_scale: 1., load_ms: 0., forward_ms: 0. },
        RemovalOutput { edit_alpha: None, rgb: vec![0.5; 6], inference_scale: 1.1, load_ms: 0., forward_ms: 0. },
        RemovalOutput { edit_alpha: None, rgb: vec![0.5; 6], inference_scale: 1., load_ms: f64::NAN, forward_ms: 0. },
        RemovalOutput { edit_alpha: None, rgb: vec![0.5; 6], inference_scale: 1., load_ms: 0., forward_ms: -1. },
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
            Ok(RemovalOutput { edit_alpha: None, rgb: input.rgb.clone(), inference_scale: 1., load_ms: 0., forward_ms: 0. })
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
    let model = crate::catalog::find("lama").unwrap();
    let path = service.path(model, true).unwrap();
    fs::write(&path, b"short").unwrap();
    let status = service.status(|_| true).into_iter().find(|status| status.id == model.id).unwrap();
    assert!(status.installed, "the manager must be able to remove a corrupt owned artifact");
    assert!(!status.available && !status.verified);
    assert!(status.reason.unwrap().contains("size"));
    service.remove(model).unwrap();
    assert!(!path.exists());
}

#[cfg(feature = "osor")]
fn osor_receipt_fixture(path: &Path) {
    use osor_rs::artifact::{BundleDigests, InputDigests, Manifest, SourceProvenance};
    fs::create_dir(path).unwrap();
    // Store-only fixture: valid tiny tensor containers, deliberately not a neural graph.
    for (index, name) in ["unet.safetensors", "vae.safetensors", "prompt.safetensors"].into_iter().enumerate() {
        let header = serde_json::to_vec(&serde_json::json!({"fixture.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0,4]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend((index as f32).to_le_bytes());
        fs::write(path.join(name), bytes).unwrap();
    }
    let provenance = SourceProvenance { repository: "https://example.invalid/source".into(), revision: "1".repeat(40), license: "fixture-only".into() };
    let digest = |name| format!("{:x}", Sha256::digest(fs::read(path.join(name)).unwrap()));
    let manifest = Manifest {
        schema_version: 1,
        upstream_revision: osor_rs::UPSTREAM_REVISION.into(),
        pipeline: osor_rs::artifact::PIPELINE.into(),
        base_model: provenance.clone(),
        vae_model: provenance.clone(),
        checkpoint: provenance.clone(),
        prompt: provenance,
        config: None,
        input_sha256: InputDigests { base_unet: "2".repeat(64), base_vae: "3".repeat(64), checkpoint: "4".repeat(64), prompt: "5".repeat(64) },
        sha256: BundleDigests { unet: digest("unet.safetensors"), vae: digest("vae.safetensors"), prompt: digest("prompt.safetensors") },
        scheduler: osor_rs::scheduler::SchedulerConfig::default(),
        timestep: 400,
        vae_scaling_factor: 0.13025,
        adapter_rank: 256,
        adapter_alpha: 256.,
    };
    fs::write(path.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
}

#[cfg(feature = "osor")]
#[test]
fn osor_reference_import_pins_local_bytes_without_copying_or_deleting_weights() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("bundle");
    osor_receipt_fixture(&source);
    let root = temp.path().join("models");
    let service = RemovalService::new(Some(root.clone()));
    service.import("osor-sdxl", &source, &NoopControl).unwrap();
    let status = service.status().into_iter().find(|s| s.id == "osor-sdxl").unwrap();
    assert!(status.installed && status.available && status.verified);
    let entries = fs::read_dir(&root).unwrap().map(|entry| entry.unwrap().file_name()).collect::<Vec<_>>();
    assert_eq!(entries, ["osor-sdxl-reference.json"]);
    let pinned = service.model_sha256("osor-sdxl").unwrap();
    assert_eq!(pinned, format!("{:x}", Sha256::digest(fs::read(source.join("manifest.json")).unwrap())));
    service.import("osor-sdxl", &source, &NoopControl).unwrap();
    let reloaded = RemovalService::new(Some(root));
    let status = reloaded.status().into_iter().find(|s| s.id == "osor-sdxl").unwrap();
    assert!(status.installed && status.available && !status.verified, "restarted service must not claim a current byte verification");
    assert_eq!(reloaded.model_sha256("osor-sdxl").unwrap(), pinned);
    service.remove("osor-sdxl").unwrap();
    assert!(source.join("unet.safetensors").is_file());
    assert!(!service.status().into_iter().find(|s| s.id == "osor-sdxl").unwrap().installed);
}

#[cfg(feature = "osor")]
#[test]
fn osor_reference_rehashes_changed_bytes_even_when_length_and_mtime_are_restored() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("bundle");
    osor_receipt_fixture(&source);
    let service = RemovalService::new(Some(temp.path().join("models")));
    service.import("osor-sdxl", &source, &NoopControl).unwrap();
    let path = source.join("unet.safetensors");
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    let mut corrupted = fs::read(&path).unwrap();
    *corrupted.last_mut().unwrap() ^= 1;
    fs::write(&path, corrupted).unwrap();
    File::open(&path).unwrap().set_modified(modified).unwrap();
    let input = RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] };
    assert!(matches!(service.infer("osor-sdxl", &input, &NoopControl), Err(Error::Integrity(_))), "changed bytes must fail before neural model load");
    let status = service.status().into_iter().find(|s| s.id == "osor-sdxl").unwrap();
    assert!(status.installed && !status.verified);
    assert!(!status.available, "failed verification must keep the corrupted bundle unavailable until a successful reimport");
}

#[cfg(all(feature = "osor", unix))]
#[test]
fn osor_reference_rejects_source_symlinks_and_cannot_remove_their_targets() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("bundle");
    osor_receipt_fixture(&source);
    let service = RemovalService::new(Some(temp.path().join("models")));
    let alias = temp.path().join("alias");
    std::os::unix::fs::symlink(&source, &alias).unwrap();
    assert!(service.import("osor-sdxl", &alias, &NoopControl).is_err());
    let victim = temp.path().join("victim");
    let expected = fs::read(source.join("unet.safetensors")).unwrap();
    fs::rename(source.join("unet.safetensors"), &victim).unwrap();
    std::os::unix::fs::symlink(&victim, source.join("unet.safetensors")).unwrap();
    assert!(service.import("osor-sdxl", &source, &NoopControl).is_err());
    service.remove("osor-sdxl").unwrap();
    assert_eq!(fs::read(&victim).unwrap(), expected);
}

#[test]
fn learned_alpha_output_is_bounded_and_matches_the_source_crop() {
    let input = RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] };
    for alpha in [vec![0.5], vec![f32::NAN, 0.5], vec![-0.1, 0.5], vec![0.5, 1.1]] {
        let output = RemovalOutput { rgb: input.rgb.clone(), edit_alpha: Some(alpha), inference_scale: 1., load_ms: 0., forward_ms: 0. };
        assert!(validate_output(&input, &output).is_err());
    }
    let output = RemovalOutput { rgb: input.rgb.clone(), edit_alpha: Some(vec![0.25, 0.75]), inference_scale: 1., load_ms: 0., forward_ms: 0. };
    assert!(validate_output(&input, &output).is_ok());
}

#[test]
fn osor_admission_includes_its_retained_learned_alpha() {
    struct Echo;
    impl RemovalBackend for Echo {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(&self, _: &str, input: &RemovalInput, _: &dyn RemovalControl) -> Result<RemovalOutput> {
            Ok(RemovalOutput { rgb: input.rgb.clone(), edit_alpha: Some(vec![0.5; input.mask.len()]), inference_scale: 1., load_ms: 0., forward_ms: 0. })
        }
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let service = RemovalService::default().with_backend(Arc::new(Echo));
    let input = RemovalInput { width: 2, height: 1, rgb: vec![0.5; 6], mask: vec![0., 1.] };
    let model_memory = u64::from(crate::catalog::find("osor-sdxl").unwrap().memory_mib) * 1024 * 1024;
    assert!(service.admit("osor-sdxl", 64, model_memory + 63).is_err());
    let mut undersized = service.admit("osor-sdxl", 56, model_memory + 56).unwrap();
    assert!(matches!(undersized.infer(&input, &NoopControl), Err(Error::InvalidInput(_))));
    drop(undersized);
    let mut permit = service.admit("osor-sdxl", 64, model_memory + 64).unwrap();
    let output = permit.infer(&input, &NoopControl).unwrap();
    let reservation = permit.into_reservation();
    assert!(matches!(service.import("osor-sdxl", Path::new("no-bundle"), &NoopControl), Err(Error::Busy)));
    drop(output);
    drop(reservation);
}

#[cfg(feature = "osor")]
#[test]
fn osor_reference_cleanup_preserves_receipt_when_staged_leaf_is_removed() {
    struct RemovePartialOnComplete(PathBuf);
    impl RemovalControl for RemovePartialOnComplete {
        fn cancelled(&self) -> bool {
            false
        }
        fn progress(&self, fraction: f32, _: &str) {
            if fraction != 1. {
                return;
            }
            let partial = fs::read_dir(&self.0)
                .unwrap()
                .find_map(|entry| {
                    let entry = entry.unwrap();
                    entry.file_name().to_string_lossy().ends_with(".partial").then(|| entry.path())
                })
                .unwrap();
            fs::remove_file(partial).unwrap();
        }
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("bundle");
    osor_receipt_fixture(&source);
    let root = temp.path().join("models");
    let service = RemovalService::new(Some(root.clone()));
    service.import("osor-sdxl", &source, &RemovePartialOnComplete(root)).unwrap();
    assert!(service.status().into_iter().find(|s| s.id == "osor-sdxl").unwrap().installed, "cleanup must retain the successfully published receipt");
}

#[test]
fn corrupt_osor_receipt_is_visible_and_removable_even_without_its_runtime() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let model = crate::catalog::find("osor-sdxl").unwrap();
    let service = NativeService::new(Some(temp.path().join("models")));
    let path = service.path(model, true).unwrap();
    fs::write(&path, b"short").unwrap();
    let status = service.status(|_| false).into_iter().find(|s| s.id == model.id).unwrap();
    assert!(status.installed && !status.available && !status.verified);
    service.remove(model).unwrap();
    assert!(!path.exists());
}

#[cfg(all(feature = "osor-metal", target_os = "macos"))]
#[test]
#[ignore = "requires an explicitly supplied local OSOR bundle and the sole real-model memory window"]
fn real_osor_service_smoke_restores_rgb_alpha_and_reuses_warm_family() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let source = PathBuf::from(std::env::var_os("PHOTOCRAFT_ML_OSOR_BUNDLE").expect("supply the local converted bundle explicitly"));
    let temp = tempfile::tempdir().unwrap();
    let service = RemovalService::new(Some(temp.path().join("models"))).with_osor_backend(crate::OsorBackend::Metal { ordinal: 0 });
    service.import("osor-sdxl", &source, &NoopControl).unwrap();
    let identity = service.model_sha256("osor-sdxl").unwrap();
    // Odd source dimensions exercise independent axis padding and exact restoration.
    let (width, height) = (101u32, 47u32);
    let mut rgb = Vec::new();
    let mut mask = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let hole = (43..58).contains(&x) && (18..30).contains(&y);
            rgb.extend(if hole { [0.8, 0.15, 0.1] } else { [0.2 + x as f32 / width as f32 * 0.3, 0.3 + y as f32 / height as f32 * 0.2, 0.4] });
            mask.push(if hole { 1. } else { 0. });
        }
    }
    let input = RemovalInput { width, height, rgb, mask };
    let transient = u64::from(width) * u64::from(height) * 32;
    let budget = u64::from(crate::catalog::find("osor-sdxl").unwrap().memory_mib) * 1024 * 1024 + transient;
    let mut cold_permit = service.admit("osor-sdxl", transient, budget).unwrap();
    let cold = cold_permit.infer(&input, &NoopControl).unwrap();
    let reservation = cold_permit.into_reservation();
    validate_output(&input, &cold).unwrap();
    assert!(cold.load_ms > 0. && cold.forward_ms > 0.);
    assert_eq!(cold.edit_alpha.as_ref().unwrap().len(), input.mask.len());
    assert!(matches!(service.remove("osor-sdxl"), Err(Error::Busy)), "returned output must retain global model ownership");
    let cold_load_ms = cold.load_ms;
    let cold_forward_ms = cold.forward_ms;
    let reference_rgb = cold.rgb.clone();
    let reference_alpha = cold.edit_alpha.clone().unwrap();
    drop(cold);
    drop(reservation);
    let mut warm_permit = service.admit("osor-sdxl", transient, budget).unwrap();
    let warm = warm_permit.infer(&input, &NoopControl).unwrap();
    let reservation = warm_permit.into_reservation();
    validate_output(&input, &warm).unwrap();
    assert_eq!(warm.load_ms, 0.);
    let rgb_max_difference = reference_rgb.iter().zip(&warm.rgb).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let alpha_max_difference = reference_alpha.iter().zip(warm.edit_alpha.as_ref().unwrap()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(rgb_max_difference <= 0.00001 && alpha_max_difference <= 0.00001, "explicit noise must reproduce the warm output");
    println!(
        "{}",
        serde_json::json!({
            "kind": "photocraft-ml OSOR service smoke", "backend": "Metal F32", "modelSha256": identity,
            "sourceWidth": width, "sourceHeight": height, "modelWidth": 104, "modelHeight": 48,
            "coldLoadMs": cold_load_ms, "coldForwardMs": cold_forward_ms, "warmLoadMs": warm.load_ms, "warmForwardMs": warm.forward_ms,
            "rgbSamples": warm.rgb.len(), "alphaSamples": warm.edit_alpha.as_ref().unwrap().len(), "inferenceScale": warm.inference_scale,
            "rgbMaxDifference": rgb_max_difference, "alphaMaxDifference": alpha_max_difference, "outputReservationBusy": true,
            "qualification": "synthetic-crop bridge smoke only; no independent reference or visual quality claim"
        })
    );
    drop(warm);
    drop(reservation);
    service.remove("osor-sdxl").unwrap();
    assert!(source.join("unet.safetensors").is_file(), "reference removal must preserve source weights");
}

#[test]
fn osor_network_install_is_rejected_before_transport_or_storage() {
    struct UnexpectedTransport(AtomicBool);
    impl ModelDownloader for UnexpectedTransport {
        fn open(&self, _: &ModelSpec) -> Result<Box<dyn Read + Send>> {
            self.0.store(true, Ordering::Release);
            Err(Error::Download("transport must remain unused".into()))
        }
    }
    let _serial = ADMISSION_TEST.lock().unwrap();
    let transport = Arc::new(UnexpectedTransport(AtomicBool::new(false)));
    let service = RemovalService::default().with_downloader(transport.clone());
    assert!(matches!(service.install("osor-sdxl", &NoopControl), Err(Error::Download(reason)) if reason.contains("local-import only")));
    assert!(!transport.0.load(Ordering::Acquire));
}

#[cfg(feature = "osor")]
#[test]
fn osor_decoded_weight_preflight_rejects_f16_expansion_above_admitted_policy() {
    let mut header = serde_json::Map::new();
    for i in 0u64..52 {
        header.insert(
            format!("extra{i:03}.weight"),
            serde_json::json!({
                "dtype": "F16", "shape": [512,262144], "data_offsets": [i * 268_435_456, (i + 1) * 268_435_456]
            }),
        );
    }
    let bytes = serde_json::to_vec(&header).unwrap();
    assert!(
        matches!(osor_reference::decoded_header_bytes(&bytes, osor_reference::MAX_DECODED_WEIGHTS), Err(Error::Integrity(reason)) if reason.contains("decoded")),
        "13GiB F16 source must not request26GiB resident F32 weights under the fixed12GiB decoded-weight policy"
    );
}

#[cfg(feature = "osor")]
#[test]
fn osor_import_rejects_sparse_oversized_decoded_weights_before_payload_hashing() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("bundle");
    osor_receipt_fixture(&source);
    let mut header = serde_json::Map::new();
    for i in 0u64..52 {
        header
            .insert(format!("extra{i:03}.weight"), serde_json::json!({"dtype":"F16", "shape":[512,262144], "data_offsets":[i*268_435_456,(i+1)*268_435_456]}));
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = File::create(source.join("unet.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + 52 * 268_435_456).unwrap();
    drop(file);
    let service = RemovalService::new(Some(temp.path().join("models")));
    assert!(
        matches!(service.import("osor-sdxl", &source, &NoopControl), Err(Error::Integrity(reason)) if reason.contains("decoded")),
        "reject oversized decoded weights from the hashed header before reading13GiB of sparse payload"
    );
    assert!(!temp.path().join("models/osor-sdxl-reference.json").exists());
    for dtype in ["F16", "BF16", "F32"] {
        let header = serde_json::to_vec(&serde_json::json!({"weight":{"dtype":dtype,"shape":[1024],"data_offsets":[0,4096]}})).unwrap();
        assert_eq!(osor_reference::decoded_header_bytes(&header, 4096).unwrap(), 4096);
        assert!(osor_reference::decoded_header_bytes(&header, 4095).is_err());
    }
}
