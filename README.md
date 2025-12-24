# Local Whisper

[🇯🇵 日本語版 README](./README.ja.md)

Local Whisper is a high-accuracy, low-latency transcription app that runs entirely on Apple Silicon (M1/M2/M3).
It captures both microphone input and system audio (e.g., web meetings, YouTube) without sending data to the cloud.

## Highlights

- 🔒 **Fully offline** – No network access required, keeping sensitive audio on your machine.
- ⚡ **Optimized for low latency** – Built on whisper.cpp to make the most of Apple Silicon’s CPU/GPU.
- 🎧 **Flexible sources** – Switch between microphone input and system audio with one tap.
- 🎥 **Screen/audio capture ready** – Record meetings or streams and transcribe them immediately.
- 🗣️ **Japanese-first experience** – Tuned for long-form Japanese speech, but works for other languages too.
- 🪄 **UI-selectable models** – Choose between base / small / medium / large v3 turbo directly from the interface.

## Requirements

- Apple Silicon Mac (macOS 13+ recommended)
- Rust 1.70 or later
- Node.js 18 or later
- pnpm
- A C++ compiler (for building whisper.cpp)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd local-whisper
   git submodule update --init --recursive
   ```

2. **Install dependencies**

   ```bash
   cd apps/desktop
   pnpm install
   ```

3. **Run in development mode**
   ```bash
   pnpm tauri dev
   ```
   Need a packaged build? Run `pnpm tauri build`.

## Model selection tips

| Model          | Characteristics                   | Suggested use case                                              |
| -------------- | --------------------------------- | --------------------------------------------------------------- |
| base           | Balanced default                  | Everyday meetings, casual videos                                |
| small          | Higher accuracy than base         | Long meeting notes                                              |
| medium         | Even higher accuracy              | Fields that require fewer transcription errors (legal, medical) |
| large v3 turbo | Highest accuracy while still fast | Subtitle generation, archival transcripts                       |

Switch models from the in-app dropdown at any time.
Pick `medium` or `large v3 turbo` when accuracy matters most; choose `base` for faster turnaround.

## How to use

1. Start the app with `pnpm tauri dev`.
2. Select the input source (**Microphone** or **System Audio**).
3. Hit **Start** to begin live transcription, and **Stop** when finished.
4. Copy results instantly or save them as logs.

## Troubleshooting

- **No audio detected**: Check macOS “System Settings → Privacy & Security → Microphone” and allow access for the app.
- **Slow performance**: Temporarily switch to a lighter model or close other CPU/GPU-intensive applications.

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tauri](https://tauri.app/)

## License

Distributed under the [MIT License](./LICENSE).
