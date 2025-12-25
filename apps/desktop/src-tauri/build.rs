use std::env;
use std::process::Command;

fn main() {
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

        let mut args = vec!["-emit-library", "-static", "-o", &lib_path];
        args.extend(swift_files.iter().map(|s| *s));
        args.extend([
            "-sdk",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
            "-target",
            "arm64-apple-macosx13.0",
        ]);

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
