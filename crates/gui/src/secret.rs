//! API キー等の秘密値を data/config.json に置くための暗号化(Windows DPAPI)。
//!
//! DPAPI はログオン中の Windows ユーザーに紐づく鍵で暗号化するため、配布物のフォルダ
//! (data/ を含む)が他人に渡っても、その PC/ユーザーでは復号できない。

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

/// 平文を暗号化して base64 文字列にする。失敗時は None(保存しない)。
pub fn protect(plain: &str) -> Option<String> {
    imp::protect(plain.as_bytes()).map(|blob| STANDARD.encode(blob))
}

/// protect() の出力を復号する。別ユーザー/別 PC で作られたもの等は None。
pub fn unprotect(encoded: &str) -> Option<String> {
    let blob = STANDARD.decode(encoded).ok()?;
    String::from_utf8(imp::unprotect(&blob)?).ok()
}

#[cfg(windows)]
mod imp {
    use windows::Win32::Foundation::{HLOCAL, LocalFree};
    use windows::Win32::Security::Cryptography::{
        CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData, CryptUnprotectData,
    };
    use windows::core::PCWSTR;

    /// DPAPI が LocalAlloc した出力をコピーして解放する。
    unsafe fn take_blob(out: CRYPT_INTEGER_BLOB) -> Vec<u8> {
        let bytes = unsafe { std::slice::from_raw_parts(out.pbData, out.cbData as usize) }.to_vec();
        unsafe { LocalFree(Some(HLOCAL(out.pbData.cast()))) };
        bytes
    }

    pub fn protect(data: &[u8]) -> Option<Vec<u8>> {
        let input = CRYPT_INTEGER_BLOB { cbData: data.len() as u32, pbData: data.as_ptr() as *mut u8 };
        let mut out = CRYPT_INTEGER_BLOB::default();
        unsafe {
            CryptProtectData(&input, PCWSTR::null(), None, None, None, CRYPTPROTECT_UI_FORBIDDEN, &mut out)
                .ok()?;
            Some(take_blob(out))
        }
    }

    pub fn unprotect(blob: &[u8]) -> Option<Vec<u8>> {
        let input = CRYPT_INTEGER_BLOB { cbData: blob.len() as u32, pbData: blob.as_ptr() as *mut u8 };
        let mut out = CRYPT_INTEGER_BLOB::default();
        unsafe {
            CryptUnprotectData(&input, None, None, None, None, CRYPTPROTECT_UI_FORBIDDEN, &mut out)
                .ok()?;
            Some(take_blob(out))
        }
    }
}

/// Windows 以外には OS の鍵保管を実装していない。保存せず、起動ごとに入力してもらう。
#[cfg(not(windows))]
mod imp {
    pub fn protect(_data: &[u8]) -> Option<Vec<u8>> {
        None
    }

    pub fn unprotect(_blob: &[u8]) -> Option<Vec<u8>> {
        None
    }
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn roundtrip() {
        let enc = super::protect("AIza-test-key").expect("protect");
        assert!(!enc.contains("AIza"));
        assert_eq!(super::unprotect(&enc).as_deref(), Some("AIza-test-key"));
    }

    #[test]
    fn garbage_is_rejected() {
        assert_eq!(super::unprotect("not-base64!!"), None);
        assert_eq!(super::unprotect("AAAA"), None);
    }
}
