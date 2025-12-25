use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_int;

extern "C" {
    fn screen_recording_start(output_path: *const c_char) -> c_int;
    fn screen_recording_stop() -> c_int;
}

pub fn start_screen_recording(output_path: &str) -> Result<(), String> {
    let c_path = CString::new(output_path).map_err(|e| e.to_string())?;

    unsafe {
        let result = screen_recording_start(c_path.as_ptr());

        let now = chrono::Local::now();
        if result == 0 {
            println!(
                "[{}] Screen recording started: {}",
                now.format("%H:%M:%S"),
                output_path
            );
            Ok(())
        } else if result == -2 {
            Err("Screen recording requires macOS 13.0+".to_string())
        } else {
            Err("Failed to start screen recording".to_string())
        }
    }
}

pub fn stop_screen_recording() -> Result<(), String> {
    unsafe {
        let result = screen_recording_stop();

        let now = chrono::Local::now();
        println!("[{}] Screen recording stopped", now.format("%H:%M:%S"));

        if result == 0 {
            Ok(())
        } else {
            Err("Failed to stop screen recording".to_string())
        }
    }
}
