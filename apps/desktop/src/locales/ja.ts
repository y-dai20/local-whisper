import type { Translations } from "./types";

const ja: Translations = {
  // Header
  clearMessages: "メッセージをクリア",
  copyAllHistory: "履歴をすべてコピー",
  copyAllHistoryDone: "履歴コピー完了",
  toggleProMode: "Pro モード切替",
  summarize: "要約",
  summarizing: "要約中...",
  summarizeSession: "今のセッションを要約",
  connecting: "接続中...",
  micOn: "マイクON",
  micOff: "マイクOFF",
  stopRecording: "録画停止",
  startRecording: "録画開始",

  // Main content
  summaryTitle: "要約",
  collapse: "折りたたむ",
  expand: "展開する",
  close: "閉じる",
  newMessages: (count: number) => `新規メッセージ ${count}件`,
  emptyMicPrompt: "マイクをオンにして、あなたの声を文字起こしします。",
  emptyRecordPrompt: "録音/録画をオンにして、システム音声を文字起こしします。",
  copied: "コピー完了",
  copy: "コピー",

  // Model setup
  installModelPrompt: "まず Whisper モデルをインストールしてください",
  installModelDescription:
    "ローカルモードではモデルがないとマイク文字起こしを開始できません。下から1つ選んでインストールしてください。",
  loadingModels: "モデル一覧を読み込み中...",
  noModelsAvailable:
    "利用可能なモデル一覧を取得できませんでした。設定画面の「モデル設定」から再読み込みしてください。",
  downloading: "ダウンロード中...",
  processing: "処理中...",
  useModel: "使用する",
  install: "インストール",

  // Settings modal
  settings: "設定",
  modelSettings: "モデル設定",
  activeModel: "使用モデル",
  currentModel: (name: string) => `使用中: ${name}`,
  availableModels: "利用可能なモデル",
  refresh: "更新",
  loading: "読み込み中...",
  noModelsFound: "利用可能なモデルが見つかりません",
  inUse: "使用中",
  use: "使用する",
  deleting: "削除中...",
  deleteLabel: "削除",
  installing: "インストール中...",

  micSettings: "マイク設定",
  inputDevice: "入力デバイス",
  defaultSuffix: " (デフォルト)",

  recordingSaveSettings: "録画・保存設定",
  saveRecording: "録画/音声を保存",
  saveFolder: "保存先フォルダ",
  saveFormatHint:
    "録画時はMP4、音声のみの場合はWAVファイルとして保存されます",
  enableScreenRecording: "画面録画を有効化",
  screenRecordingHint:
    "有効時は録画ボタンで画面+音声を録画、無効時は音声のみ保存",

  whisperModelSettings: "Whisper モデル設定",
  saving: "保存中...",
  save: "保存",
  contextLength: (value: number) => `コンテキスト長 (audio_ctx: ${value})`,
  contextLengthHint:
    "長くするほど過去の音声を参照できますが、計算量とメモリ使用量が増えます。",
  temperature: (value: string) => `温度 (temperature: ${value})`,
  temperatureHint:
    "数値を上げると出力が多様になります。0に近いほど安定した結果になります。",

  streamingSettings: "ストリーミング設定",
  vadThreshold: (value: string) => `VAD 閾値 (${value})`,
  vadThresholdHint:
    "数値が低いほど小さな声でも検知しやすく、高いほど大きな音しか検知しなくなります。",
  transcriptionInterval: "文字起こし間隔 (秒)",
  transcriptionIntervalHint:
    "短くすると小刻みに更新され、長くするとまとまった文章で届きます。",

  micPermissionRequired: "マイクの許可が必要です",
  initializing: "初期化中...",

  // Error messages
  streamingConfigError: (msg: string) => `ストリーミング設定エラー: ${msg}`,
  whisperConfigError: (msg: string) => `Whisper設定エラー: ${msg}`,
  backendConfigError: (msg: string) =>
    `文字起こしバックエンド設定エラー: ${msg}`,
  enableRecordingSaveFirst:
    "録画保存を有効にして保存先フォルダを設定してください",
  initializeModelFirst: "録画を開始する前にモデルを初期化してください",
  recordingStartError: (msg: string) => `録画開始エラー: ${msg}`,
  recordingStopError: (msg: string) => `録画停止エラー: ${msg}`,
  screenRecordingConfigError: (msg: string) => `画面録画設定エラー: ${msg}`,
  modelScanError: (msg: string) => `モデルスキャンエラー: ${msg}`,
  remoteModelError: (msg: string) => `リモートモデル取得エラー: ${msg}`,
  deviceListError: (msg: string) => `マイクデバイス取得エラー: ${msg}`,
  deviceSelectError: (msg: string) => `デバイス選択エラー: ${msg}`,
  languageChangeError: (msg: string) => `言語変更エラー: ${msg}`,
  modelInstallError: (msg: string) => `モデルインストールエラー: ${msg}`,
  modelDeleteError: (msg: string) => `モデル削除エラー: ${msg}`,
  initError: (msg: string) => `初期化エラー: ${msg}`,
  modelInitializing: "モデルを初期化中です...",
  micStartError: (msg: string) => `マイク開始エラー: ${msg}`,
  micStopError: (msg: string) => `録音停止エラー: ${msg}`,
  transcriptionResumeError: (msg: string) => `文字起こし再開エラー: ${msg}`,
  transcriptionStopError: (msg: string) => `文字起こし停止エラー: ${msg}`,
  copyError: (msg: string) => `コピーエラー: ${msg}`,
  historyCopyError: (msg: string) => `履歴コピーエラー: ${msg}`,
  audioPlaybackError: (msg: string) => `音声再生エラー: ${msg}`,
  recordingSaveConfigError: (msg: string) => `録画保存設定エラー: ${msg}`,
  noSessionToSummarize: "要約対象のセッションがありません",
  noTextToSummarize: "要約対象のテキストがありません",
  summaryError: (msg: string) => `要約エラー: ${msg}`,
};

export default ja;
