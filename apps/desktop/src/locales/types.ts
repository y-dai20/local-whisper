export interface Translations {
  // Header
  clearMessages: string;
  copyAllHistory: string;
  copyAllHistoryDone: string;
  toggleProMode: string;
  summarize: string;
  summarizing: string;
  summarizeSession: string;
  connecting: string;
  micOn: string;
  micOff: string;
  stopRecording: string;
  startRecording: string;

  // Main content
  summaryTitle: string;
  collapse: string;
  expand: string;
  close: string;
  newMessages: (count: number) => string;
  emptyMicPrompt: string;
  emptyRecordPrompt: string;
  copied: string;
  copy: string;

  // Model setup
  installModelPrompt: string;
  installModelDescription: string;
  loadingModels: string;
  noModelsAvailable: string;
  downloading: string;
  processing: string;
  useModel: string;
  install: string;

  // Settings modal
  settings: string;
  modelSettings: string;
  activeModel: string;
  currentModel: (name: string) => string;
  availableModels: string;
  refresh: string;
  loading: string;
  noModelsFound: string;
  inUse: string;
  use: string;
  deleting: string;
  deleteLabel: string;
  installing: string;

  micSettings: string;
  inputDevice: string;
  defaultSuffix: string;

  recordingSaveSettings: string;
  saveRecording: string;
  saveFolder: string;
  saveFormatHint: string;
  enableScreenRecording: string;
  screenRecordingHint: string;

  whisperModelSettings: string;
  saving: string;
  save: string;
  contextLength: (value: number) => string;
  contextLengthHint: string;
  temperature: (value: string) => string;
  temperatureHint: string;

  streamingSettings: string;
  vadThreshold: (value: string) => string;
  vadThresholdHint: string;
  transcriptionInterval: string;
  transcriptionIntervalHint: string;

  micPermissionRequired: string;
  initializing: string;

  // Error messages
  streamingConfigError: (msg: string) => string;
  whisperConfigError: (msg: string) => string;
  backendConfigError: (msg: string) => string;
  enableRecordingSaveFirst: string;
  initializeModelFirst: string;
  recordingStartError: (msg: string) => string;
  recordingStopError: (msg: string) => string;
  screenRecordingConfigError: (msg: string) => string;
  modelScanError: (msg: string) => string;
  remoteModelError: (msg: string) => string;
  deviceListError: (msg: string) => string;
  deviceSelectError: (msg: string) => string;
  languageChangeError: (msg: string) => string;
  modelInstallError: (msg: string) => string;
  modelDeleteError: (msg: string) => string;
  initError: (msg: string) => string;
  modelInitializing: string;
  micStartError: (msg: string) => string;
  micStopError: (msg: string) => string;
  transcriptionResumeError: (msg: string) => string;
  transcriptionStopError: (msg: string) => string;
  copyError: (msg: string) => string;
  historyCopyError: (msg: string) => string;
  audioPlaybackError: (msg: string) => string;
  recordingSaveConfigError: (msg: string) => string;
  noSessionToSummarize: string;
  noTextToSummarize: string;
  summaryError: (msg: string) => string;
}
