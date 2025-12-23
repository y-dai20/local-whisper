# Local Whisper

Apple Silicon (M1/M2/M3) 向けに最適化した、完全ローカル動作の高精度・低遅延文字起こしアプリです。
Web 会議や YouTube などのシステム音声も、マイク入力も、そのままテキスト化できます。


https://github.com/user-attachments/assets/402253cf-a4e6-493c-95c7-a45b916474e2


## なにができる？

- 🔒 **完全ローカル** – ネットワークアクセス不要。機密音声も安心。
- ⚡ **低遅延** – whisper.cpp ベースの最適化で、Apple Silicon の CPU/GPU を無駄なく活用。
- 🎧 **音声ソースを選べる** – マイク入力とシステム音声入力をワンタップで切り替え。
- 🎥 **録画もできる** – Web 会議や配信をキャプチャして、そのまま文字起こしに活用。
- 🗣️ **日本語に強い** – 長時間の会議記録や動画も安定して文字起こし。
- 🪄 **モデルを UI から選択** – base / small / medium / large v3 turbo を切り替えて精度と速度を調整。

## 必要環境

- Apple Silicon 搭載 Mac（macOS 13 以上推奨）
- Rust 1.70+
- Node.js 18+
- pnpm
- C++ コンパイラ（whisper.cpp のビルド用）

## セットアップ

1. **リポジトリを取得**

   ```bash
   git clone <repository-url>
   cd local-whisper
   git submodule update --init --recursive
   ```

2. **依存関係をインストール**

   ```bash
   cd apps/desktop
   pnpm install
   ```

3. **開発モードで起動**
   ```bash
   pnpm tauri dev
   ```
   ビルド済みバイナリが欲しい場合は `pnpm tauri build` を利用してください。

## モデル選択のヒント

| モデル         | 特徴                   | 想定用途                          |
| -------------- | ---------------------- | --------------------------------- |
| base           | バランス型。初期設定。 | 日常的な会議や動画視聴            |
| small          | base より高精度        | 長時間の会議メモ                  |
| medium         | さらに高精度           | 医療/法律など誤差を減らしたい場面 |
| large v3 turbo | 最高精度ながら高速     | 字幕生成やアーカイブ用途          |

モデルはアプリ内のプルダウンから即時切り替えできます。
精度重視なら `medium / large v3 turbo`、レスポンス優先なら `base` を選ぶのがおすすめです。

## 使い方

1. `pnpm tauri dev` でアプリを起動。
2. 入力ソースを **マイク** か **システム音声** から選択。
3. [開始] でリアルタイム文字起こしを開始。必要に応じて [停止]。
4. テキストはその場でコピーしたり、ログとして保存できます。

## トラブルシューティング

- **音声が取得できない**: macOS の「システム設定 > プライバシーとセキュリティ > マイク」でアプリの権限を許可してください。
- **パフォーマンスが出ない**: 一時的に別モデルへ切り替え、または他の重いアプリを終了して CPU/GPU 負荷を下げてください。

## 謝辞

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tauri](https://tauri.app/)

## ライセンス

本プロジェクトは [MIT License](./LICENSE) で提供されています。
