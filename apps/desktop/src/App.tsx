import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Mic, MicOff, Play, Square } from "lucide-react";
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
      if (models.length > 0) {
        setSelectedModel(models[0].path);
      }
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
        const result = await invoke<{
          success: boolean;
          text?: string;
          error?: string;
        }>("transcribe_audio", {
          audioData: audioData,
          language: selectedLanguage === "auto" ? null : selectedLanguage,
        });

        if (result.success && result.text) {
          const segment: TranscriptionSegment = {
            text: result.text,
            timestamp: Date.now(),
            audioData: audioData,
          };
          setTranscriptions((prev) => [...prev, segment]);
        }
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
    <div className="flex flex-col h-screen w-screen">
      {/* Header */}
      <header className="bg-[#1a1a2e] border-b border-slate-700 px-8 py-4 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex justify-between items-center gap-8">
          <h1 className="text-2xl font-bold bg-linear-to-r from-primary to-secondary bg-clip-text text-transparent whitespace-nowrap">
            🎙️ Local Whisper
          </h1>

          <div className="flex items-center gap-6 flex-wrap">
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                モデル
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-2 bg-[#0f0f23] border border-slate-700 rounded-md text-slate-200 text-sm cursor-pointer transition-all hover:border-primary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 min-w-[150px]"
              >
                {availableModels.map((model) => (
                  <option key={model.path} value={model.path}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                言語
              </label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                disabled={!isMuted || isTranscribing}
                className="px-3 py-2 bg-[#0f0f23] border border-slate-700 rounded-md text-slate-200 text-sm cursor-pointer transition-all hover:border-primary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 min-w-[150px] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {availableLanguages.map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                マイク
              </label>
              <select
                value={selectedAudioDevice}
                onChange={(e) => handleAudioDeviceChange(e.target.value)}
                disabled={!isMuted || isTranscribing}
                className="px-3 py-2 bg-[#0f0f23] border border-slate-700 rounded-md text-slate-200 text-sm cursor-pointer transition-all hover:border-primary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 min-w-[200px] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {audioDevices.map((device) => (
                  <option key={device.name} value={device.name}>
                    {device.name} {device.is_default ? "(デフォルト)" : ""}
                  </option>
                ))}
              </select>
            </div>

            {hasMicPermission === false && (
              <div className="px-4 py-2 rounded-full text-xs font-semibold uppercase tracking-wider bg-red-500/10 text-red-500 border border-red-500">
                ⚠️ マイク権限なし
              </div>
            )}
            {hasMicPermission === true && !isInitialized && (
              <div className="px-4 py-2 rounded-full text-xs font-semibold uppercase tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500">
                初期化中...
              </div>
            )}
            {hasMicPermission === true && isInitialized && (
              <div className="px-4 py-2 rounded-full text-xs font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-500 border border-emerald-500">
                準備完了
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto px-8 py-8 flex flex-col items-center">
        {error && (
          <div className="w-full max-w-3xl px-6 py-4 bg-red-500/10 border border-red-500 rounded-lg text-red-500 mb-6 text-sm">
            ⚠️ {error}
          </div>
        )}

        <div className="w-full max-w-3xl flex-1 flex flex-col">
          {transcriptions.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-400">
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
                <div
                  key={index}
                  className="bg-[#1a1a2e] rounded-2xl p-5 border border-slate-700 transition-all hover:border-primary hover:shadow-lg hover:shadow-primary/15 transcription-slide-in"
                >
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-xs text-slate-400 font-semibold">
                      {new Date(segment.timestamp).toLocaleTimeString()}
                    </span>
                    {segment.audioData && (
                      <button
                        className="w-7 h-7 rounded-full bg-linear-to-br from-primary to-secondary flex items-center justify-center transition-all hover:scale-110 hover:shadow-lg hover:shadow-primary/40 active:scale-95"
                        onClick={() => playAudio(segment.audioData!, index)}
                        title={playingIndex === index ? "停止" : "音声を再生"}
                      >
                        {playingIndex === index ? (
                          <Square className="w-3 h-3 text-white fill-white" />
                        ) : (
                          <Play className="w-3 h-3 text-white fill-white" />
                        )}
                      </button>
                    )}
                  </div>
                  <p className="text-base leading-relaxed text-slate-200">
                    {segment.text}
                  </p>
                </div>
              ))}
            </div>
          )}

          {isTranscribing && (
            <div className="flex items-center justify-center gap-3 p-4 text-slate-400 text-sm">
              <div className="w-5 h-5 border-2 border-slate-700 border-t-primary rounded-full spinner"></div>
              <span>文字起こし中...</span>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="flex-shrink-0 px-8 py-8 flex justify-center items-center bg-[#1a1a2e] border-t border-slate-700">
        <button
          className={`relative w-16 h-16 rounded-full border-none cursor-pointer transition-all flex flex-col items-center justify-center gap-2 ${
            !isInitialized || isTranscribing
              ? "opacity-50 cursor-not-allowed bg-slate-700"
              : isMuted
              ? "bg-linear-to-br from-primary to-secondary shadow-lg shadow-primary/30 hover:scale-105 hover:shadow-xl hover:shadow-primary/40 active:scale-95"
              : "bg-linear-to-br from-red-500 to-red-600 mic-button-pulse"
          }`}
          onClick={toggleMute}
          disabled={!isInitialized || isTranscribing}
        >
          <div className="text-white flex items-center justify-center">
            {isMuted ? (
              <MicOff className="w-6 h-6" />
            ) : (
              <Mic className="w-6 h-6" />
            )}
          </div>
        </button>
      </footer>
    </div>
  );
}

export default App;
