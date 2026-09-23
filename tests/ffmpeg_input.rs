#![cfg(target_os = "windows")]

use std::ffi::OsString;
use std::sync::Arc;

use anyhow::Result;
use ffmpeg_the_third::{self as ffmpeg, format};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/video/keyframes.mp4")
}

#[test]
fn owned_interrupt_survives_probe_and_input_move_then_drops() -> Result<()> {
    ffmpeg::init()?;
    let capture = Arc::new(());
    let weak = Arc::downgrade(&capture);
    let mut options = ffmpeg::Dictionary::new();
    options.set("protocol_whitelist", "file");
    options.set("probesize", "5242880");
    let mut probe_options = ffmpeg::Dictionary::new();
    probe_options.set("threads", "1");
    let input = format::input_with_dictionary_and_interrupt(fixture(), options, &probe_options, {
        let capture = Arc::clone(&capture);
        move || {
            let _ = &capture;
            false
        }
    })?;
    drop(capture);
    assert!(
        weak.upgrade().is_some(),
        "input must retain its callback after probing"
    );
    std::thread::spawn(move || {
        assert!(input.streams().count() > 0);
        drop(input);
    })
    .join()
    .expect("input worker panicked");
    assert!(
        weak.upgrade().is_none(),
        "closing input must release callback capture"
    );
    Ok(())
}

#[test]
fn failed_open_and_invalid_path_release_interrupt_capture() -> Result<()> {
    ffmpeg::init()?;
    let missing = fixture().with_file_name("absent-video-for-ffmpeg-input-test.mp4");
    for path in [
        missing.into_os_string(),
        OsString::from("invalid\0path.mp4"),
    ] {
        let capture = Arc::new(());
        let weak = Arc::downgrade(&capture);
        let result = format::input_with_dictionary_and_interrupt(
            &path,
            ffmpeg::Dictionary::new(),
            &ffmpeg::Dictionary::new(),
            {
                let capture = Arc::clone(&capture);
                move || {
                    let _ = &capture;
                    false
                }
            },
        );
        assert!(result.is_err());
        if path.to_string_lossy().contains('\0') {
            assert!(matches!(result, Err(ffmpeg::Error::Other { errno }) if errno == libc::EINVAL));
        }
        drop(capture);
        assert!(
            weak.upgrade().is_none(),
            "failed open must release callback capture"
        );
    }
    let temp = tempfile::tempdir()?;
    let corrupt = temp.path().join("corrupt.mp4");
    std::fs::write(&corrupt, b"not a video container")?;
    let capture = Arc::new(());
    let weak = Arc::downgrade(&capture);
    let result = format::input_with_interrupt(&corrupt, {
        let capture = Arc::clone(&capture);
        move || {
            let _ = &capture;
            false
        }
    });
    assert!(result.is_err());
    drop(capture);
    assert!(
        weak.upgrade().is_none(),
        "corrupt input must release callback capture"
    );
    Ok(())
}

#[test]
fn interrupt_cancels_fixture_open_or_probe() -> Result<()> {
    ffmpeg::init()?;
    let capture = Arc::new(());
    let weak = Arc::downgrade(&capture);
    let result = format::input_with_interrupt(fixture(), {
        let capture = Arc::clone(&capture);
        move || {
            let _ = &capture;
            true
        }
    });
    assert!(
        matches!(result, Err(ffmpeg::Error::Exit)),
        "an always-true interrupt must propagate FFmpeg's AVERROR_EXIT"
    );
    drop(capture);
    assert!(
        weak.upgrade().is_none(),
        "cancelled input must release callback capture"
    );
    Ok(())
}
