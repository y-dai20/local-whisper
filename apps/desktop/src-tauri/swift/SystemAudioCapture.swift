import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation

@available(macOS 12.3, *)
class SystemAudioCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    private var stream: SCStream?
    private var audioCallback: (@convention(c) (UnsafePointer<Float>, Int) -> Void)?
    private var isCapturing = false
    private var hasLoggedFormatInfo = false

    func startCapture(callback: @escaping @convention(c) (UnsafePointer<Float>, Int) -> Void) async throws {
        self.audioCallback = callback

        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        guard let display = content.displays.first else {
            throw NSError(
                domain: "com.local-whisper.system-audio",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No displays found for capture"]
            )
        }

        print("Display found: \(display)")

        // システム音声をキャプチャするには、displayとexcludingWindowsを使用
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.excludesCurrentProcessAudio = false
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: self)

        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "com.local-whisper.audio"))

        try await stream.startCapture()

        self.stream = stream
        self.isCapturing = true
        self.hasLoggedFormatInfo = false
        print("System audio stream started successfully")
    }

    func stopCapture() async {
        guard let stream = self.stream else { return }

        do {
            try await stream.stopCapture()
        } catch {
            print("Failed to stop capture: \(error)")
        }

        self.stream = nil
        self.isCapturing = false
        self.audioCallback = nil
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        if #available(macOS 13.0, *) {
            guard type == .audio, let callback = self.audioCallback else { return }

            guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

            var length: Int = 0
            var dataPointer: UnsafeMutablePointer<Int8>?

            let status = CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)

            guard status == kCMBlockBufferNoErr, let data = dataPointer else { return }

            guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
            guard let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else { return }

            let channelCount = Int(asbd.pointee.mChannelsPerFrame)
            let bytesPerSample = Int(asbd.pointee.mBitsPerChannel / 8)
            let frameCount = length / (channelCount * bytesPerSample)

            var monoSamples = [Float](repeating: 0, count: frameCount)

            if bytesPerSample == 4 {
                let floatData = data.withMemoryRebound(to: Float.self, capacity: length / 4) { $0 }

                for i in 0..<frameCount {
                    var sum: Float = 0
                    for ch in 0..<channelCount {
                        sum += floatData[i * channelCount + ch]
                    }
                    monoSamples[i] = sum / Float(channelCount)
                }
            } else if bytesPerSample == 2 {
                let int16Data = data.withMemoryRebound(to: Int16.self, capacity: length / 2) { $0 }

                for i in 0..<frameCount {
                    var sum: Float = 0
                    for ch in 0..<channelCount {
                        sum += Float(int16Data[i * channelCount + ch]) / Float(Int16.max)
                    }
                    monoSamples[i] = sum / Float(channelCount)
                }
            }

            if !hasLoggedFormatInfo {
                let maxSample = monoSamples.max(by: { abs($0) < abs($1) }).map { abs($0) } ?? 0
                print(
                    """
                    [SystemAudioCapture] format info:
                      sampleRate=\(asbd.pointee.mSampleRate)
                      channelCount=\(channelCount)
                      bytesPerSample=\(bytesPerSample)
                      frameCount=\(frameCount)
                      maxAbsSample=\(String(format: "%.6f", maxSample))
                    """
                )
                hasLoggedFormatInfo = true
            }

            monoSamples.withUnsafeBufferPointer { buffer in
                callback(buffer.baseAddress!, frameCount)
            }
        }
    }
}

private var captureInstance: SystemAudioCapture?
private var lastSystemAudioError: NSString = ""

@_cdecl("system_audio_start")
public func systemAudioStart(callback: @escaping @convention(c) (UnsafePointer<Float>, Int) -> Void) -> Int32 {
    if #available(macOS 13.0, *) {
        let semaphore = DispatchSemaphore(value: 0)
        var startResult: Int32 = -1

        Task.detached {
            do {
                let capture = SystemAudioCapture()
                try await capture.startCapture(callback: callback)
                captureInstance = capture
                lastSystemAudioError = ""
                startResult = 0
            } catch {
                let nsError = error as NSError
                lastSystemAudioError = error.localizedDescription as NSString
                startResult = Int32(nsError.code)
            }
            semaphore.signal()
        }

        semaphore.wait()
        return startResult
    } else {
        lastSystemAudioError = "System audio capture requires macOS 13.0+" as NSString
        return -2
    }
}

@_cdecl("system_audio_stop")
public func systemAudioStop() -> Int32 {
    if #available(macOS 13.0, *) {
        guard let capture = captureInstance else {
            return -1
        }

        Task {
            await capture.stopCapture()
        }

        captureInstance = nil
        return 0
    } else {
        return -2
    }
}

@_cdecl("system_audio_last_error")
public func systemAudioLastError() -> UnsafePointer<CChar>? {
    return lastSystemAudioError.utf8String
}
