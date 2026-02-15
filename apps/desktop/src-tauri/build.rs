use std::env;
use std::process::Command;

fn deployment_major(version: &str) -> u32 {
    version
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
}

fn main() {
    env_logger::init();

    #[cfg(target_os = "macos")]
    {
        let swift_files = vec![
            "swift/SystemAudioCapture.swift",
            "swift/ScreenRecorder.swift",
        ];
        let out_dir = env::var("OUT_DIR").unwrap();
        let lib_name = "libSwiftBridge.a";
        let lib_path = format!("{}/{}", out_dir, lib_name);

        for file in &swift_files {
            println!("cargo:rerun-if-changed={}", file);
        }
        println!("cargo:rerun-if-env-changed=SWIFT_TARGET");
        println!("cargo:rerun-if-env-changed=SWIFT_SDK_PATH");
        println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

        let rust_target = env::var("TARGET").expect("TARGET is not set");
        let swift_arch = if rust_target.starts_with("aarch64-apple-darwin") {
            "arm64"
        } else if rust_target.starts_with("x86_64-apple-darwin") {
            "x86_64"
        } else {
            panic!("unsupported target for swift bridge: {rust_target}");
        };
        let deployment_target_raw =
            env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "13.0".to_string());
        let deployment_target = if deployment_major(&deployment_target_raw) < 13
        {
            "13.0".to_string()
        } else {
            deployment_target_raw
        };
        let default_swift_target = format!("{swift_arch}-apple-macosx{deployment_target}");
        let swift_target = env::var("SWIFT_TARGET").unwrap_or(default_swift_target);

        let sdk_path = env::var("SWIFT_SDK_PATH").unwrap_or_else(|_| {
            let output = Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("Failed to get macOS SDK path via xcrun");
            if !output.status.success() {
                eprintln!("xcrun failed:");
                eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
                eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
                panic!("Failed to get macOS SDK path via xcrun");
            }
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        });

        let mut args = vec!["-emit-library", "-static", "-o", &lib_path];
        args.extend(swift_files.iter().map(|s| *s));
        args.extend(["-sdk", &sdk_path, "-target", &swift_target]);

        let output = Command::new("swiftc")
            .args(&args)
            .output()
            .expect("Failed to compile Swift code");

        if !output.status.success() {
            eprintln!("Swift compilation failed:");
            eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("Swift compilation failed");
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=SwiftBridge");
        println!("cargo:rustc-link-lib=framework=ScreenCaptureKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/System/Library/Frameworks");
    }

    tauri_build::build()
}
