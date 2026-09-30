//! Document portal paths remain the source of truth for filesystem access. The host
//! spelling is presentation-only: it may be inaccessible inside the Flatpak sandbox.

use camino::{Utf8Path, Utf8PathBuf};

pub fn host_path(path: &Utf8Path) -> Option<Utf8PathBuf> {
    #[cfg(target_os = "linux")]
    {
        std::env::var_os("FLATPAK_ID")?;
        resolve(path, read_host_path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn resolve(
    path: &Utf8Path,
    read: impl Fn(&Utf8Path) -> Option<Utf8PathBuf>,
) -> Option<Utf8PathBuf> {
    // Older portals expose the attribute on the exported directory only. Append
    // the relative suffix without canonicalizing it against the host filesystem.
    for ancestor in path.ancestors() {
        // This is a Linux path even when portable resolver tests run on Windows.
        if let Some(host) = read(ancestor).filter(|host| host.as_str().starts_with('/')) {
            return Some(host.join(path.strip_prefix(ancestor).ok()?));
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_host_path(path: &Utf8Path) -> Option<Utf8PathBuf> {
    let path = std::ffi::CString::new(path.as_str()).ok()?;
    let mut bytes = vec![0_u8; 4096];
    // SAFETY: path and attribute are NUL-terminated; the output buffer is writable
    // for the exact length passed. A missing/unsupported attribute is a normal fallback.
    let size = unsafe {
        libc::getxattr(
            path.as_ptr(),
            c"user.document-portal.host-path".as_ptr(),
            bytes.as_mut_ptr().cast(),
            bytes.len(),
        )
    };
    let size = usize::try_from(size).ok()?;
    bytes.truncate(size);
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    let host = String::from_utf8(bytes).ok()?;
    if host.contains('\0') {
        return None;
    }
    Some(Utf8PathBuf::from(host))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn reads_real_directory_xattr_and_retains_descendant_suffix() {
        let directory = tempfile::tempdir().unwrap();
        let root = Utf8Path::from_path(directory.path()).unwrap();
        let name = std::ffi::CString::new(root.as_str()).unwrap();
        let host = b"/media/photos/Album";
        // SAFETY: the path/attribute are NUL-terminated and host is readable for
        // the supplied length. The temporary directory is owned by this test.
        let result = unsafe {
            libc::setxattr(
                name.as_ptr(),
                c"user.document-portal.host-path".as_ptr(),
                host.as_ptr().cast(),
                host.len(),
                0,
            )
        };
        assert_eq!(result, 0);
        assert_eq!(
            read_host_path(root),
            Some(Utf8PathBuf::from("/media/photos/Album"))
        );
        assert_eq!(
            resolve(&root.join("trip/photo.jpg"), read_host_path),
            Some(Utf8PathBuf::from("/media/photos/Album/trip/photo.jpg"))
        );
    }

    #[test]
    fn translates_directory_descendants_without_changing_access_path() {
        let path = Utf8Path::new("/run/user/1000/doc/abc/Photos/trips/cat.jpg");
        let host = resolve(path, |ancestor| {
            (ancestor == Utf8Path::new("/run/user/1000/doc/abc/Photos"))
                .then(|| Utf8PathBuf::from("/media/photos/Photos"))
        });
        assert_eq!(
            host.as_deref(),
            Some(Utf8Path::new("/media/photos/Photos/trips/cat.jpg"))
        );
        assert_eq!(path.as_str(), "/run/user/1000/doc/abc/Photos/trips/cat.jpg");
    }

    #[test]
    fn missing_or_invalid_attributes_preserve_the_portal_fallback() {
        let path = Utf8Path::new("/run/user/1000/doc/abc/cat.jpg");
        assert_eq!(resolve(path, |_| None), None);
        assert_eq!(resolve(path, |_| Some(Utf8PathBuf::from("relative"))), None);
        assert_eq!(
            resolve(path, |p| (p == path)
                .then(|| Utf8PathBuf::from("/home/cat.jpg"))),
            Some(Utf8PathBuf::from("/home/cat.jpg"))
        );
    }
}
