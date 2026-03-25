import type { Translations } from "./types";

const en: Translations = {
  // Header
  clearMessages: "Clear messages",
  copyAllHistory: "Copy all history",
  copyAllHistoryDone: "History copied",
  toggleProMode: "Toggle Pro mode",
  summarize: "Summarize",
  summarizing: "Summarizing...",
  summarizeSession: "Summarize current session",
  connecting: "Connecting...",
  micOn: "Mic ON",
  micOff: "Mic OFF",
  stopRecording: "Stop recording",
  startRecording: "Start recording",

  // Main content
  summaryTitle: "Summary",
  collapse: "Collapse",
  expand: "Expand",
  close: "Close",
  newMessages: (count: number) => `${count} new message${count === 1 ? "" : "s"}`,
  emptyMicPrompt: "Turn on the microphone to transcribe your voice.",
  emptyRecordPrompt:
    "Start recording to transcribe system audio.",
  copied: "Copied",
  copy: "Copy",

  // Model setup
  installModelPrompt: "Install a Whisper model to get started",
  installModelDescription:
    "Local mode requires a model to transcribe audio. Choose one below to install.",
  loadingModels: "Loading models...",
  noModelsAvailable:
    "Could not load available models. Try reloading from Model Settings.",
  downloading: "Downloading...",
  processing: "Processing...",
  useModel: "Use",
  install: "Install",

  // Settings modal
  settings: "Settings",
  modelSettings: "Model Settings",
  activeModel: "Active Model",
  currentModel: (name: string) => `Current: ${name}`,
  availableModels: "Available Models",
  refresh: "Refresh",
  loading: "Loading...",
  noModelsFound: "No models found",
  inUse: "In use",
  use: "Use",
  deleting: "Deleting...",
  deleteLabel: "Delete",
  installing: "Installing...",

  micSettings: "Microphone Settings",
  inputDevice: "Input Device",
  defaultSuffix: " (default)",

  recordingSaveSettings: "Recording & Save Settings",
  saveRecording: "Save recordings",
  saveFolder: "Save folder",
  saveFormatHint: "Recordings are saved as MP4 (with video) or WAV (audio only)",
  enableScreenRecording: "Enable screen recording",
  screenRecordingHint:
    "When enabled, the record button captures screen + audio; otherwise audio only",

  whisperModelSettings: "Whisper Model Settings",
  saving: "Saving...",
  save: "Save",
  contextLength: (value: number) => `Context length (audio_ctx: ${value})`,
  contextLengthHint:
    "Longer values reference more past audio but increase computation and memory usage.",
  temperature: (value: string) => `Temperature: ${value}`,
  temperatureHint:
    "Higher values produce more varied output. Values closer to 0 give more stable results.",

  streamingSettings: "Streaming Settings",
  vadThreshold: (value: string) => `VAD Threshold (${value})`,
  vadThresholdHint:
    "Lower values detect quieter speech; higher values require louder audio.",
  transcriptionInterval: "Transcription interval (sec)",
  transcriptionIntervalHint:
    "Shorter intervals update more frequently; longer intervals produce fuller sentences.",

  micPermissionRequired: "Microphone permission is required",
  initializing: "Initializing...",

  // Error messages
  streamingConfigError: (msg: string) => `Streaming config error: ${msg}`,
  whisperConfigError: (msg: string) => `Whisper config error: ${msg}`,
  backendConfigError: (msg: string) => `Transcription backend error: ${msg}`,
  enableRecordingSaveFirst:
    "Enable recording save and set a save folder first",
  initializeModelFirst: "Initialize a model before starting recording",
  recordingStartError: (msg: string) => `Recording start error: ${msg}`,
  recordingStopError: (msg: string) => `Recording stop error: ${msg}`,
  screenRecordingConfigError: (msg: string) =>
    `Screen recording config error: ${msg}`,
  modelScanError: (msg: string) => `Model scan error: ${msg}`,
  remoteModelError: (msg: string) => `Remote model error: ${msg}`,
  deviceListError: (msg: string) => `Audio device error: ${msg}`,
  deviceSelectError: (msg: string) => `Device selection error: ${msg}`,
  languageChangeError: (msg: string) => `Language change error: ${msg}`,
  modelInstallError: (msg: string) => `Model install error: ${msg}`,
  modelDeleteError: (msg: string) => `Model delete error: ${msg}`,
  initError: (msg: string) => `Initialization error: ${msg}`,
  modelInitializing: "Model is initializing...",
  micStartError: (msg: string) => `Mic start error: ${msg}`,
  micStopError: (msg: string) => `Mic stop error: ${msg}`,
  transcriptionResumeError: (msg: string) =>
    `Transcription resume error: ${msg}`,
  transcriptionStopError: (msg: string) =>
    `Transcription stop error: ${msg}`,
  copyError: (msg: string) => `Copy error: ${msg}`,
  historyCopyError: (msg: string) => `History copy error: ${msg}`,
  audioPlaybackError: (msg: string) => `Audio playback error: ${msg}`,
  recordingSaveConfigError: (msg: string) =>
    `Recording save config error: ${msg}`,
  noSessionToSummarize: "No session to summarize",
  noTextToSummarize: "No text to summarize",
  summaryError: (msg: string) => `Summary error: ${msg}`,
};

export default en;
