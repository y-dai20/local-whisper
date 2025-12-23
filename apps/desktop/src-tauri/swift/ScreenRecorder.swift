import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

@available(macOS 13.0, *)
class ScreenRecorder: NSObject, SCStreamDelegate, SCStreamOutput {
    private var stream: SCStream?
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var microphoneInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var isRecording = false
    private var outputURL: URL?
    private var startTime: CMTime?

    func startRecording(outputPath: String) async throws {
        let url = URL(fileURLWithPath: outputPath)
        self.outputURL = url

        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        guard let display = content.displays.first else {
            throw NSError(domain: "ScreenRecorder", code: 1, userInfo: [NSLocalizedDescriptionKey: "No displays found"])
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.width = display.width
        config.height = display.height
        config.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        config.capturesAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        config.excludesCurrentProcessAudio = false

        // マイク音声もキャプチャ (macOS 15.0+)
        if #available(macOS 15.0, *) {
            config.captureMicrophone = true
        }

        let assetWriter = try AVAssetWriter(url: url, fileType: .mp4)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: display.width,
            AVVideoHeightKey: display.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 5_000_000
            ]
        ]

        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = true

        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: display.width,
            kCVPixelBufferHeightKey as String: display.height
        ]

        let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        self.pixelBufferAdaptor = pixelBufferAdaptor

        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 48000,
            AVNumberOfChannelsKey: 2,
            AVEncoderBitRateKey: 128000
        ]

        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioInput.expectsMediaDataInRealTime = true

        // マイク用の別のオーディオ入力
        let microphoneInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        microphoneInput.expectsMediaDataInRealTime = true

        if assetWriter.canAdd(videoInput) {
            assetWriter.add(videoInput)
        }

        if assetWriter.canAdd(audioInput) {
            assetWriter.add(audioInput)
        }

        if assetWriter.canAdd(microphoneInput) {
            assetWriter.add(microphoneInput)
        }

        self.assetWriter = assetWriter
        self.videoInput = videoInput
        self.audioInput = audioInput
        self.microphoneInput = microphoneInput

        let stream = SCStream(filter: filter, configuration: config, delegate: self)

        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue(label: "com.local-whisper.screen"))
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "com.local-whisper.recording-audio"))

        // マイク音声の出力ストリームを追加 (macOS 15.0+)
        if #available(macOS 15.0, *) {
            try stream.addStreamOutput(self, type: .microphone, sampleHandlerQueue: DispatchQueue(label: "com.local-whisper.recording-microphone"))
        }

        try await stream.startCapture()

        self.stream = stream
        self.isRecording = true

        print("[ScreenRecorder] Recording started: \(outputPath)")
    }

    func stopRecording() async throws {
        guard let stream = self.stream else { return }

        try await stream.stopCapture()

        self.stream = nil
        self.isRecording = false

        if let videoInput = self.videoInput {
            videoInput.markAsFinished()
        }

        if let audioInput = self.audioInput {
            audioInput.markAsFinished()
        }

        if let microphoneInput = self.microphoneInput {
            microphoneInput.markAsFinished()
        }

        if let assetWriter = self.assetWriter {
            await assetWriter.finishWriting()

            if assetWriter.status == .completed {
                if let url = self.outputURL {
                    print("[ScreenRecorder] Recording saved: \(url.path)")
                }
            } else if let error = assetWriter.error {
                print("[ScreenRecorder] Recording failed: \(error)")
            }
        }

        self.assetWriter = nil
        self.videoInput = nil
        self.audioInput = nil
        self.microphoneInput = nil
        self.pixelBufferAdaptor = nil
        self.startTime = nil
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard isRecording else { return }

        if startTime == nil {
            startTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            assetWriter?.startWriting()
            assetWriter?.startSession(atSourceTime: startTime!)
        }

        switch type {
        case .screen:
            if let videoInput = self.videoInput,
               let adaptor = self.pixelBufferAdaptor,
               videoInput.isReadyForMoreMediaData {

                guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    return
                }

                let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                adaptor.append(imageBuffer, withPresentationTime: presentationTime)
            }
        case .audio:
            if let audioInput = self.audioInput, audioInput.isReadyForMoreMediaData {
                audioInput.append(sampleBuffer)
            }
        case .microphone:
            // マイク音声は専用のトラックに保存
            if let microphoneInput = self.microphoneInput, microphoneInput.isReadyForMoreMediaData {
                microphoneInput.append(sampleBuffer)
            }
        @unknown default:
            break
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        print("[ScreenRecorder] Stream stopped with error: \(error)")
    }
}

private var recorderInstance: ScreenRecorder?

@_cdecl("screen_recording_start")
public func screenRecordingStart(outputPath: UnsafePointer<CChar>) -> Int32 {
    if #available(macOS 13.0, *) {
        let path = String(cString: outputPath)
        let recorder = ScreenRecorder()

        Task {
            do {
                try await recorder.startRecording(outputPath: path)
                recorderInstance = recorder
            } catch {
                print("[ScreenRecorder] Failed to start recording: \(error)")
            }
        }

        return 0
    } else {
        print("[ScreenRecorder] Screen recording requires macOS 13.0+")
        return -2
    }
}

@_cdecl("screen_recording_stop")
public func screenRecordingStop() -> Int32 {
    if #available(macOS 13.0, *) {
        guard let recorder = recorderInstance else {
            return -1
        }

        Task {
            do {
                try await recorder.stopRecording()
                recorderInstance = nil
            } catch {
                print("[ScreenRecorder] Failed to stop recording: \(error)")
            }
        }

        return 0
    } else {
        return -2
    }
}
