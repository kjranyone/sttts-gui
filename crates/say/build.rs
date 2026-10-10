//! Windows リソース(バージョン情報)を exe に埋め込む。
//! バージョン情報は Cargo.toml から生成する(SignPath の署名は ProductName / ProductVersion の一致を確かめる)。

fn main() {
    #[cfg(target_os = "windows")]
    {
        let rc = resource("", "sttts-say", "sttts-say: Irodori-TTS lines without the GUI");
        let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("app.rc");
        std::fs::write(&path, rc).unwrap();
        embed_resource::compile(&path, embed_resource::NONE).manifest_optional().unwrap();
    }
}

/// VERSIONINFO を含むリソーススクリプト。ProductName はプロジェクト名(sttts-gui)で全 exe 共通
#[cfg(target_os = "windows")]
fn resource(extra: &str, internal: &str, description: &str) -> String {
    let v = |k: &str| std::env::var(k).unwrap();
    let (major, minor, patch, version) = (v("CARGO_PKG_VERSION_MAJOR"), v("CARGO_PKG_VERSION_MINOR"), v("CARGO_PKG_VERSION_PATCH"), v("CARGO_PKG_VERSION"));
    format!(
        r#"{extra}
1 VERSIONINFO
FILEVERSION {major},{minor},{patch},0
PRODUCTVERSION {major},{minor},{patch},0
FILEOS 0x40004
FILETYPE 0x1
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904b0"
    BEGIN
      VALUE "FileDescription", "{description}"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "{internal}"
      VALUE "OriginalFilename", "{internal}.exe"
      VALUE "LegalCopyright", "Copyright (c) 2026 Kojiro Tanaka. MIT License."
      VALUE "ProductName", "sttts-gui"
      VALUE "ProductVersion", "{version}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#
    )
}
