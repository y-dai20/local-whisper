use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GGML_METAL_PATH_RESOURCES");
    println!("cargo:rustc-env=GGML_METAL_USE_FAST_FP16=1");

    if env::var("GGML_METAL_PATH_RESOURCES").is_ok() {
        // User explicitly configured the path; propagate it at compile time as well.
        println!(
            "cargo:rustc-env=GGML_METAL_PATH_RESOURCES={}",
            env::var("GGML_METAL_PATH_RESOURCES").unwrap()
        );
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let ggml_path = manifest_dir.join("../../vendor/whisper.cpp/ggml/src");
    let default_path = ggml_path.join("ggml-metal");

    if let Ok(metadata) = fs::metadata(&default_path) {
        if metadata.is_dir() {
            if let Ok(canonical) = default_path.canonicalize() {
                println!(
                    "cargo:rustc-env=GGML_METAL_PATH_RESOURCES={}",
                    canonical.display()
                );
            }
        }
    }
}
