//! Windows リソース(アイコンとバージョン情報)を exe に埋め込む。
//! バージョン情報は Cargo.toml から生成する(SignPath の署名は ProductName / ProductVersion の一致を確かめる)。

fn main() {
    // build.rs の cfg はビルドする側の OS なので、ターゲットは環境変数で見る
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // canonicalize は \\?\ 付きになり rc.exe が読めないので、そのまま / 区切りにする
    let icon = manifest.join("../../assets/app-icon.ico").display().to_string().replace('\\', "/");
    println!("cargo:rerun-if-changed={icon}");
    let rc = resource(&format!("1 ICON \"{icon}\"\n"), "sttts-gui", "sttts-gui: voice chat with Irodori-TTS");
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("app.rc");
    std::fs::write(&path, rc).unwrap();
    embed_resource::compile(&path, embed_resource::NONE).manifest_optional().unwrap();
}

/// VERSIONINFO を含むリソーススクリプト。ProductName はプロジェクト名(sttts-gui)で全 exe 共通
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
