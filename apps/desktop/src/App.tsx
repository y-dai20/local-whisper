import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Mic, MicOff, Play, Square, Settings } from "lucide-react";
import "./App.css";

interface TranscriptionSegment {
  text: string;
  timestamp: number;
  audioData?: number[];
}

interface ModelInfo {
  name: string;
  path: string;
  size: number;
}

interface RemoteModelStatus {
  id: string;
  name: string;
  filename: string;
  size: number;
  description: string;
  installed: boolean;
  path?: string;
}

interface AudioDevice {
  name: string;
  is_default: boolean;
}

function App() {
  const [isMuted, setIsMuted] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedLanguage, setSelectedLanguage] = useState<string>("ja");
  const [availableLanguages, setAvailableLanguages] = useState<
    [string, string][]
  >([]);
  const [transcriptions, setTranscriptions] = useState<TranscriptionSegment[]>(
    []
  );
  const [error, setError] = useState("");
  const [playingIndex, setPlayingIndex] = useState<number | null>(null);
  const [currentAudioSource, setCurrentAudioSource] =
    useState<AudioBufferSourceNode | null>(null);
  const [currentAudioContext, setCurrentAudioContext] =
    useState<AudioContext | null>(null);
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [selectedAudioDevice, setSelectedAudioDevice] = useState<string>("");
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(
    null
  );
  const [showSettings, setShowSettings] = useState(false);
  const [remoteModels, setRemoteModels] = useState<RemoteModelStatus[]>([]);
  const [modelOperations, setModelOperations] = useState<
    Record<string, boolean>
  >({});
  const [isLoadingRemoteModels, setIsLoadingRemoteModels] = useState(false);

  useEffect(() => {
    const unlisten = listen<TranscriptionSegment>(
      "transcription-segment",
      (event) => {
        setTranscriptions((prev) => [...prev, event.payload]);
      }
    );

    scanForModels();
    loadLanguages();
    checkMicPermission();
    loadAudioDevices();
    loadRemoteModels();

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    if (selectedModel) {
      setIsInitialized(false);
      initializeWhisper();
    }
  }, [selectedModel]);

  const scanForModels = async () => {
    try {
      const models = await invoke<ModelInfo[]>("scan_models");
      setAvailableModels(models);
      setSelectedModel((prev) => {
        if (prev && models.some((model) => model.path === prev)) {
          return prev;
        }
        return models[0]?.path ?? "";
      });
    } catch (err) {
      console.error("Model scan error:", err);
      setError(`モデルスキャンエラー: ${err}`);
    }
  };

  const loadLanguages = async () => {
    try {
      const langs = await invoke<[string, string][]>("get_supported_languages");
      setAvailableLanguages(langs);
    } catch (err) {
      console.error("Failed to load languages:", err);
    }
  };

  const loadRemoteModels = async () => {
    try {
      setIsLoadingRemoteModels(true);
      const models = await invoke<RemoteModelStatus[]>("list_remote_models");
      setRemoteModels(models);
    } catch (err) {
      console.error("Failed to load remote models:", err);
      setError(`リモートモデル取得エラー: ${err}`);
    } finally {
      setIsLoadingRemoteModels(false);
    }
  };

  const loadAudioDevices = async () => {
    try {
      const devices = await invoke<AudioDevice[]>("list_audio_devices");
      setAudioDevices(devices);
      const preferredDevice =
        devices.find((d) => d.is_default) ??
        (devices.length > 0 ? devices[0] : undefined);
      if (preferredDevice) {
        setSelectedAudioDevice(preferredDevice.name);
        await invoke("select_audio_device", {
          deviceName: preferredDevice.name,
        });
      }
    } catch (err) {
      console.error("Failed to load audio devices:", err);
      setError(`マイクデバイス取得エラー: ${err}`);
    }
  };

  const checkMicPermission = async () => {
    try {
      const hasPermission = await invoke<boolean>(
        "check_microphone_permission"
      );
      setHasMicPermission(hasPermission);
    } catch (err) {
      console.error("Failed to check mic permission:", err);
      setHasMicPermission(false);
    }
  };

  const handleAudioDeviceChange = async (deviceName: string) => {
    try {
      await invoke("select_audio_device", { deviceName });
      setSelectedAudioDevice(deviceName);
    } catch (err) {
      console.error("Failed to select audio device:", err);
      setError(`デバイス選択エラー: ${err}`);
    }
  };

  const refreshAllModels = async () => {
    await scanForModels();
    await loadRemoteModels();
  };

  const handleInstallModel = async (modelId: string) => {
    setModelOperations((prev) => ({ ...prev, [modelId]: true }));
    try {
      await invoke<ModelInfo>("install_model", { modelId });
      await refreshAllModels();
    } catch (err) {
      console.error("Install model error:", err);
      setError(
        `モデルインストールエラー: ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    } finally {
      setModelOperations((prev) => ({ ...prev, [modelId]: false }));
    }
  };

  const handleDeleteModel = async (model: RemoteModelStatus) => {
    if (!model.path) return;
    setModelOperations((prev) => ({ ...prev, [model.id]: true }));
    try {
      await invoke("delete_model", { modelPath: model.path });
      setSelectedModel((prev) => (prev === model.path ? "" : prev));
      await refreshAllModels();
    } catch (err) {
      console.error("Delete model error:", err);
      setError(
        `モデル削除エラー: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setModelOperations((prev) => ({ ...prev, [model.id]: false }));
    }
  };

  const formatModelSize = (bytes: number) => {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const initializeWhisper = async () => {
    if (!selectedModel) return;

    try {
      setError("");
      await invoke("initialize_whisper", { modelPath: selectedModel });
      setIsInitialized(true);
    } catch (err) {
      setError(`初期化エラー: ${err}`);
      setIsInitialized(false);
    }
  };

  const toggleMute = async () => {
    console.log("toggleMute", isMuted);
    if (isMuted) {
      await startRecording();
    } else {
      await stopRecording();
    }
  };

  const startRecording = async () => {
    if (!isInitialized) {
      setError("モデルを初期化中です...");
      console.log("startRecording", isInitialized);
      return;
    }

    try {
      await invoke("start_recording");
      setIsMuted(false);
      setError("");
    } catch (err) {
      console.error("Recording start error:", err);
      setError(
        `録音開始エラー: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  };

  const stopRecording = async () => {
    setIsMuted(true);
    setIsTranscribing(true);

    try {
      const audioData = await invoke<number[]>("stop_recording");

      if (audioData.length > 0) {
        await invoke<{
          success: boolean;
          text?: string;
          error?: string;
        }>("transcribe_audio", {
          audioData: audioData,
          language: selectedLanguage === "auto" ? null : selectedLanguage,
        });
        // Backend will emit "transcription-segment" event, no need to add manually
      }
    } catch (err) {
      console.error("Recording stop error:", err);
      setError(
        `録音停止エラー: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setIsTranscribing(false);
    }
  };

  const stopAudio = () => {
    if (currentAudioSource) {
      currentAudioSource.stop();
      currentAudioSource.disconnect();
      setCurrentAudioSource(null);
    }
    if (currentAudioContext) {
      currentAudioContext.close();
      setCurrentAudioContext(null);
    }
    setPlayingIndex(null);
  };

  const playAudio = (audioData: number[], index: number) => {
    try {
      if (playingIndex === index) {
        stopAudio();
        return;
      }

      if (currentAudioSource || currentAudioContext) {
        stopAudio();
      }

      const audioContext = new AudioContext({ sampleRate: 16000 });
      const audioBuffer = audioContext.createBuffer(1, audioData.length, 16000);
      const channelData = audioBuffer.getChannelData(0);

      for (let i = 0; i < audioData.length; i++) {
        channelData[i] = audioData[i];
      }

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);

      setCurrentAudioSource(source);
      setCurrentAudioContext(audioContext);
      setPlayingIndex(index);

      source.onended = () => {
        audioContext.close();
        setCurrentAudioSource(null);
        setCurrentAudioContext(null);
        setPlayingIndex(null);
      };

      source.start(0);
    } catch (err) {
      console.error("Audio playback error:", err);
      setError(
        `音声再生エラー: ${err instanceof Error ? err.message : String(err)}`
      );
      setPlayingIndex(null);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-base-100">
      {/* Header */}
      <header className="flex bg-base-200 border-b border-base-300 px-4 py-1">
        <div className="flex-1"></div>
        <button
          className="btn btn-ghost btn-circle btn-sm"
          onClick={() => setShowSettings(true)}
        >
          <Settings className="w-5 h-5" />
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto space-y-3">
          {error && (
            <div className="alert alert-error">
              <span className="text-sm">{error}</span>
            </div>
          )}

          {transcriptions.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-base-content/50">
              <div className="text-6xl mb-4 opacity-50">🎤</div>
              <p className="text-lg mb-2">
                マイクボタンをクリックして録音を開始
              </p>
              <p className="text-sm opacity-70">
                録音を停止すると自動で文字起こしが実行されます
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-6 pb-8">
              {transcriptions.map((segment, index) => (
                <div key={index} className="card bg-base-200 shadow-sm">
                  <div className="card-body p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <p className="text-sm leading-relaxed">
                          {segment.text}
                        </p>
                        <p className="text-xs opacity-60 mt-2">
                          {new Date(segment.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                      {segment.audioData && (
                        <button
                          onClick={() => playAudio(segment.audioData!, index)}
                          className="btn btn-circle btn-sm btn-ghost"
                        >
                          {playingIndex === index ? (
                            <Square className="w-4 h-4" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {isTranscribing && (
            <div className="flex items-center justify-center gap-3 py-4">
              <span className="loading loading-spinner loading-md"></span>
              <span className="text-sm">文字起こし中...</span>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-base-200 border-t border-base-300 h-20 flex items-center">
        <div className="flex items-center justify-between w-full px-4">
          <div className="flex-1"></div>
          <button
            className={`btn btn-circle btn-lg ${
              !isInitialized || isTranscribing
                ? "btn-disabled"
                : isMuted
                ? "btn-primary"
                : "btn-error animate-pulse"
            }`}
            onClick={toggleMute}
            disabled={!isInitialized || isTranscribing}
          >
            {isMuted ? (
              <MicOff className="w-6 h-6" />
            ) : (
              <Mic className="w-6 h-6" />
            )}
          </button>
          <div className="flex-1 flex justify-end items-center">
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="select select-bordered select-sm w-32"
            >
              {availableLanguages.length === 0 ? (
                <option value="ja">日本語</option>
              ) : (
                availableLanguages.map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </footer>

      {/* Settings Modal */}
      {showSettings && (
        <dialog className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-4">設定</h3>

            <div className="space-y-4">
              <div className="form-control">
                <label className="label">
                  <span className="label-text">モデル</span>
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="select select-bordered w-full"
                >
                  {availableModels.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="label-text font-semibold">
                    利用可能なモデル
                  </span>
                  <button
                    className="btn btn-xs"
                    onClick={refreshAllModels}
                    disabled={isLoadingRemoteModels}
                  >
                    更新
                  </button>
                </div>

                <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
                  {isLoadingRemoteModels ? (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="loading loading-spinner loading-xs"></span>
                      読み込み中...
                    </div>
                  ) : remoteModels.length === 0 ? (
                    <p className="text-sm opacity-60">
                      利用可能なモデルが見つかりません
                    </p>
                  ) : (
                    remoteModels.map((model) => (
                      <div
                        key={model.id}
                        className="border border-base-300 rounded-xl p-3 flex flex-col gap-2"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="font-medium text-sm">{model.name}</p>
                            <p className="text-xs opacity-70">
                              {model.description}
                            </p>
                            <p className="text-xs opacity-60 mt-1">
                              {formatModelSize(model.size)}
                            </p>
                            {model.installed && model.path && (
                              <p className="text-[11px] opacity-50 mt-1 break-all">
                                {model.path}
                              </p>
                            )}
                          </div>
                          <div className="flex flex-col gap-2 items-end">
                            {model.installed ? (
                              <>
                                <button
                                  className="btn btn-xs btn-outline"
                                  disabled={
                                    selectedModel === model.path ||
                                    !model.path ||
                                    modelOperations[model.id]
                                  }
                                  onClick={() =>
                                    model.path && setSelectedModel(model.path)
                                  }
                                >
                                  {selectedModel === model.path
                                    ? "使用中"
                                    : "このモデルを使用"}
                                </button>
                                <button
                                  className="btn btn-xs btn-error"
                                  onClick={() => handleDeleteModel(model)}
                                  disabled={modelOperations[model.id]}
                                >
                                  {modelOperations[model.id]
                                    ? "削除中..."
                                    : "削除"}
                                </button>
                              </>
                            ) : (
                              <button
                                className="btn btn-xs btn-primary"
                                onClick={() => handleInstallModel(model.id)}
                                disabled={modelOperations[model.id]}
                              >
                                {modelOperations[model.id]
                                  ? "インストール中..."
                                  : "インストール"}
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="form-control">
                <label className="label">
                  <span className="label-text">マイク</span>
                </label>
                <select
                  value={selectedAudioDevice}
                  onChange={(e) => handleAudioDeviceChange(e.target.value)}
                  className="select select-bordered w-full"
                >
                  {audioDevices.map((device) => (
                    <option key={device.name} value={device.name}>
                      {device.name}
                      {device.is_default ? " (デフォルト)" : ""}
                    </option>
                  ))}
                </select>
              </div>

              {hasMicPermission === false && (
                <div className="alert alert-warning">
                  <span className="text-xs">⚠️ マイクの許可が必要です</span>
                </div>
              )}

              {!isInitialized && selectedModel && (
                <div className="alert alert-info">
                  <span className="text-xs">初期化中...</span>
                </div>
              )}
            </div>

            <div className="modal-action">
              <button className="btn" onClick={() => setShowSettings(false)}>
                閉じる
              </button>
            </div>
          </div>
          <form method="dialog" className="modal-backdrop">
            <button onClick={() => setShowSettings(false)}>close</button>
          </form>
        </dialog>
      )}
    </div>
  );
}

export default App;
