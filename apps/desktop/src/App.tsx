import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  Mic,
  MicOff,
  Play,
  Square,
  Settings,
  Circle,
  StopCircle,
  Copy,
  Check,
  Trash2,
  Moon,
  Sun,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  X,
} from "lucide-react";
import "./App.css";

interface TranscriptionSegment {
  text: string;
  timestamp: number;
  audioData?: number[];
  sessionId: number;
  messageId: number;
  isFinal: boolean;
  source: string;
}

interface SessionTranscription {
  sessionKey: string;
  sessionId: number;
  source: string;
  messages: TranscriptionSegment[];
  audioChunks: Record<number, number[]>;
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

interface StreamingConfig {
  vadThreshold: number;
  partialIntervalSeconds: number;
}

interface VoiceActivityEvent {
  source: string;
  isActive: boolean;
  sessionId: number;
  timestamp: number;
}

type VoiceActivitySourceState = {
  isActive: boolean;
  sessionId: number | null;
};

type VoiceActivityState = Record<"user" | "system", VoiceActivitySourceState>;

type VoiceSource = "user" | "system";

interface WhisperParamsConfig {
  audioCtx: number;
  temperature: number;
}

interface TranscriptionBackendConfig {
  mode: "local" | "api";
}

interface SummaryResponse {
  summary?: string;
  text?: string;
  result?: string;
}

interface BackendErrorEvent {
  message: string;
  fallbackMode?: "local" | "api";
}

const API_BASE_URL =
  (import.meta.env.VITE_ASR_API_BASE_URL as string | undefined)?.trim() || "";

function App() {
  const [isMuted, setIsMuted] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedLanguage, setSelectedLanguage] = useState<string>("ja");
  const [availableLanguages, setAvailableLanguages] = useState<
    [string, string][]
  >([]);
  const [transcriptions, setTranscriptions] = useState<SessionTranscription[]>(
    [],
  );
  const [error, setError] = useState("");
  const [playingSessionKey, setPlayingSessionKey] = useState<string | null>(
    null,
  );
  const [copiedSessionKey, setCopiedSessionKey] = useState<string | null>(null);
  const [currentAudioSource, setCurrentAudioSource] =
    useState<AudioBufferSourceNode | null>(null);
  const [currentAudioContext, setCurrentAudioContext] =
    useState<AudioContext | null>(null);
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [selectedAudioDevice, setSelectedAudioDevice] = useState<string>("");
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(
    null,
  );
  const [showSettings, setShowSettings] = useState(false);
  const [remoteModels, setRemoteModels] = useState<RemoteModelStatus[]>([]);
  const [modelOperations, setModelOperations] = useState<
    Record<string, boolean>
  >({});
  const [isLoadingRemoteModels, setIsLoadingRemoteModels] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const [streamingConfig, setStreamingConfig] = useState<StreamingConfig>({
    vadThreshold: 0.1,
    partialIntervalSeconds: 2,
  });
  const [isSavingStreamingConfig, setIsSavingStreamingConfig] = useState(false);
  const [whisperParams, setWhisperParams] = useState<WhisperParamsConfig>({
    audioCtx: 1000,
    temperature: 0,
  });
  const [isSavingWhisperParams, setIsSavingWhisperParams] = useState(false);
  const [recordingSaveEnabled, setRecordingSaveEnabled] = useState(false);
  const [recordingSavePath, setRecordingSavePath] = useState("");
  const [screenRecordingEnabled, setScreenRecordingEnabled] = useState(false);
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [isRecordingBusy, setIsRecordingBusy] = useState(false);
  const [isMicBusy, setIsMicBusy] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summaryPanelText, setSummaryPanelText] = useState<string | null>(null);
  const [isSummaryExpanded, setIsSummaryExpanded] = useState(true);
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "light");
  const [transcriptionMode, setTranscriptionMode] = useState<"local" | "api">(
    "local",
  );
  const [copiedAllHistory, setCopiedAllHistory] = useState(false);
  const [voiceActivity, setVoiceActivity] = useState<VoiceActivityState>({
    user: { isActive: false, sessionId: null },
    system: { isActive: false, sessionId: null },
  });
  const [newMessageCount, setNewMessageCount] = useState(0);
  const [isAutoScrollPaused, setIsAutoScrollPaused] = useState(false);
  const transcriptionSuppressedForPlaybackRef = useRef(false);
  const copyFeedbackTimeoutRef = useRef<number | null>(null);
  const copyAllFeedbackTimeoutRef = useRef<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const previousMessageCountRef = useRef(0);
  const previousScrollHeightRef = useRef(0);

  const upsertTranscriptionSegment = useCallback(
    (segment: TranscriptionSegment) => {
      setTranscriptions((prev) => {
        const sessionKey = `${segment.source}-${segment.sessionId}`;
        const sessions = [...prev];
        const sessionIndex = sessions.findIndex((s) => s.sessionKey === sessionKey);

        const upsertMessages = (existing: TranscriptionSegment[] | undefined) => {
          if (!existing) {
            return [segment];
          }
          const messages = [...existing];
          const messageIndex = messages.findIndex((m) => m.messageId === segment.messageId);
          if (messageIndex >= 0) {
            messages[messageIndex] = segment;
          } else {
            messages.push(segment);
          }
          messages.sort((a, b) => a.messageId - b.messageId);
          return messages;
        };

        const upsertAudioChunks = (existing: Record<number, number[]> | undefined) => {
          const chunks = { ...(existing || {}) };
          if (segment.audioData?.length) {
            chunks[segment.messageId] = segment.audioData;
          }
          return chunks;
        };

        if (sessionIndex >= 0) {
          const session = sessions[sessionIndex];
          sessions[sessionIndex] = {
            ...session,
            messages: upsertMessages(session.messages),
            audioChunks: upsertAudioChunks(session.audioChunks),
          };
          return sessions;
        }

        return [
          ...sessions,
          {
            sessionKey,
            sessionId: segment.sessionId,
            source: segment.source,
            messages: [segment],
            audioChunks: segment.audioData?.length
              ? { [segment.messageId]: segment.audioData }
              : {},
          },
        ];
      });
    },
    [],
  );

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const loadStreamingConfig = useCallback(async () => {
    try {
      const config = await invoke<StreamingConfig>("get_streaming_config");
      setStreamingConfig(config);
      localStorage.setItem("vadThreshold", config.vadThreshold.toString());
      localStorage.setItem(
        "partialIntervalSeconds",
        config.partialIntervalSeconds.toString(),
      );
    } catch (err) {
      console.error("Failed to load streaming config:", err);
    }
  }, []);

  const saveStreamingConfig = useCallback(
    async (config: StreamingConfig) => {
      setIsSavingStreamingConfig(true);
      try {
        await invoke("set_streaming_config", {
          config,
        });
        setStreamingConfig(config);
        localStorage.setItem("vadThreshold", config.vadThreshold.toString());
        if (config.partialIntervalSeconds) {
          localStorage.setItem(
            "partialIntervalSeconds",
            config.partialIntervalSeconds.toString(),
          );
        }
      } catch (err) {
        console.error("Failed to save streaming config:", err);
        setError(
          `ストリーミング設定エラー: ${
            err instanceof Error ? err.message : String(err)
          }`,
        );
      } finally {
        setIsSavingStreamingConfig(false);
      }
    },
    [setError],
  );

  const loadWhisperParams = useCallback(async () => {
    try {
      const params = await invoke<WhisperParamsConfig>("get_whisper_params");
      setWhisperParams(params);
      localStorage.setItem("whisperAudioCtx", params.audioCtx.toString());
      localStorage.setItem("whisperTemperature", params.temperature.toString());
    } catch (err) {
      console.error("Failed to load Whisper params:", err);
    }
  }, []);

  const saveWhisperParams = useCallback(
    async (params: WhisperParamsConfig) => {
      setIsSavingWhisperParams(true);
      try {
        await invoke("set_whisper_params", {
          config: params,
        });
        setWhisperParams(params);
        localStorage.setItem("whisperAudioCtx", params.audioCtx.toString());
        localStorage.setItem(
          "whisperTemperature",
          params.temperature.toString(),
        );
      } catch (err) {
        console.error("Failed to save Whisper params:", err);
        setError(
          `Whisper設定エラー: ${
            err instanceof Error ? err.message : String(err)
          }`,
        );
      } finally {
        setIsSavingWhisperParams(false);
      }
    },
    [setError],
  );

  const loadTranscriptionBackendConfig = useCallback(async () => {
    try {
      const config = await invoke<TranscriptionBackendConfig>(
        "get_transcription_backend_config",
      );
      setTranscriptionMode(config.mode);
    } catch (err) {
      console.error("Failed to load transcription backend config:", err);
    }
  }, []);

  const saveTranscriptionBackendConfig = useCallback(
    async (config: TranscriptionBackendConfig) => {
      try {
        const normalizedConfig: TranscriptionBackendConfig = {
          mode: config.mode,
        };
        await invoke("set_transcription_backend_config", {
          config: normalizedConfig,
        });
        setTranscriptionMode(normalizedConfig.mode);
      } catch (err) {
        console.error("Failed to save transcription backend config:", err);
        setError(
          `文字起こしバックエンド設定エラー: ${
            err instanceof Error ? err.message : String(err)
          }`,
        );
      }
    },
    [setError],
  );

  const toggleRecording = async () => {
    if (isRecordingBusy) return;

    if (!isRecordingActive) {
      if (!recordingSaveEnabled || !recordingSavePath) {
        setError("録画保存を有効にして保存先フォルダを設定してください");
        return;
      }
      if (
        transcriptionMode === "local" &&
        !screenRecordingEnabled &&
        !isInitialized
      ) {
        setError("録画を開始する前にモデルを初期化してください");
        return;
      }

      // 既存の文字起こし履歴をクリアして新しい録画用にリセット
      setTranscriptions([]);
      setPlayingSessionKey(null);
      if (currentAudioSource) {
        currentAudioSource.stop();
        currentAudioSource.disconnect();
        setCurrentAudioSource(null);
      }
      if (currentAudioContext) {
        currentAudioContext.close();
        setCurrentAudioContext(null);
      }

      setIsRecordingBusy(true);
      let startedScreenRecording = false;
      let startedRecording = false;
      let startedSystemAudio = false;
      try {
        if (screenRecordingEnabled) {
          await invoke("start_screen_recording");
          startedScreenRecording = true;
        }
        await invoke("start_recording", {
          language: selectedLanguage === "auto" ? null : selectedLanguage,
        });
        startedRecording = true;
        await invoke("start_system_audio");
        startedSystemAudio = true;
        setIsRecordingActive(true);
        setError("");
      } catch (err) {
        console.error("Failed to start recording session:", err);
        if (startedSystemAudio) {
          try {
            await invoke("stop_system_audio");
          } catch (stopErr) {
            console.error("Failed to rollback system audio:", stopErr);
          }
        }
        if (startedRecording) {
          try {
            await invoke("stop_recording");
          } catch (stopErr) {
            console.error("Failed to rollback recording session:", stopErr);
          }
        }
        if (startedScreenRecording) {
          try {
            await invoke("stop_screen_recording");
          } catch (stopErr) {
            console.error("Failed to rollback screen recording:", stopErr);
          }
        }
        setIsRecordingActive(false);
        setError(
          `録画開始エラー: ${err instanceof Error ? err.message : String(err)}`,
        );
      } finally {
        setIsRecordingBusy(false);
      }
      return;
    }

    setIsRecordingBusy(true);
    try {
      if (screenRecordingEnabled) {
        await invoke("stop_screen_recording");
      }
      await invoke("stop_recording");
      await invoke("stop_system_audio");
      setIsRecordingActive(false);
      setError("");
    } catch (err) {
      console.error("Failed to stop recording session:", err);
      setError(
        `録画停止エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsRecordingBusy(false);
    }
  };

  const loadRecordingSaveConfig = async () => {
    try {
      const [enabled, path] = await invoke<[boolean, string | null]>(
        "get_recording_save_config",
      );
      setRecordingSaveEnabled(enabled);
      setRecordingSavePath(path || "");
    } catch (err) {
      console.error("Failed to load recording save config:", err);
    }
  };

  const saveRecordingSaveConfig = async (enabled: boolean, path: string) => {
    try {
      await invoke("set_recording_save_config", {
        enabled,
        path: path || null,
      });
      setRecordingSaveEnabled(enabled);
      setRecordingSavePath(path);
      localStorage.setItem("recordingSaveEnabled", enabled.toString());
      localStorage.setItem("recordingSavePath", path);
    } catch (err) {
      console.error("Failed to save recording save config:", err);
      setError(
        `録画保存設定エラー: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
    }
  };

  const loadSettingsFromLocalStorage = () => {
    try {
      const savedModelPath = localStorage.getItem("selectedModelPath");
      if (savedModelPath) {
        setSelectedModel(savedModelPath);
      }

      const savedLanguage = localStorage.getItem("selectedLanguage");
      if (savedLanguage) {
        setSelectedLanguage(savedLanguage);
      }

      const savedVadThreshold = localStorage.getItem("vadThreshold");
      const savedPartialInterval = localStorage.getItem(
        "partialIntervalSeconds",
      );
      if (savedVadThreshold || savedPartialInterval) {
        setStreamingConfig((prev) => ({
          vadThreshold: savedVadThreshold
            ? parseFloat(savedVadThreshold)
            : prev.vadThreshold,
          partialIntervalSeconds: savedPartialInterval
            ? parseFloat(savedPartialInterval)
            : prev.partialIntervalSeconds,
        }));
      }

      const savedAudioCtx = localStorage.getItem("whisperAudioCtx");
      const savedTemperature = localStorage.getItem("whisperTemperature");
      if (savedAudioCtx || savedTemperature) {
        setWhisperParams((prev) => {
          const updated = {
            audioCtx: savedAudioCtx
              ? parseInt(savedAudioCtx, 10)
              : prev.audioCtx,
            temperature: savedTemperature
              ? parseFloat(savedTemperature)
              : prev.temperature,
          };
          invoke("set_whisper_params", { config: updated }).catch((err) =>
            console.error("Failed to reapply Whisper params:", err),
          );
          return updated;
        });
      }

      const savedRecordingSaveEnabled = localStorage.getItem(
        "recordingSaveEnabled",
      );
      const savedRecordingSavePath = localStorage.getItem("recordingSavePath");
      if (savedRecordingSaveEnabled !== null) {
        const enabled = savedRecordingSaveEnabled === "true";
        const path = savedRecordingSavePath || "";
        setRecordingSaveEnabled(enabled);
        setRecordingSavePath(path);
        saveRecordingSaveConfig(enabled, path);
      }

      const savedScreenRecordingEnabled = localStorage.getItem(
        "screenRecordingEnabled",
      );
      if (savedScreenRecordingEnabled !== null) {
        const enabled = savedScreenRecordingEnabled === "true";
        setScreenRecordingEnabled(enabled);
        saveScreenRecordingConfig(enabled);
      }
    } catch (err) {
      console.error("Failed to load settings from localStorage:", err);
    }
  };

  const loadScreenRecordingConfig = async () => {
    try {
      const enabled = await invoke<boolean>("get_screen_recording_config");
      setScreenRecordingEnabled(enabled);
    } catch (err) {
      console.error("Failed to load screen recording config:", err);
    }
  };

  const saveScreenRecordingConfig = async (enabled: boolean) => {
    try {
      await invoke("set_screen_recording_config", {
        enabled,
      });
      setScreenRecordingEnabled(enabled);
      localStorage.setItem("screenRecordingEnabled", enabled.toString());
    } catch (err) {
      console.error("Failed to save screen recording config:", err);
      setError(
        `画面録画設定エラー: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
    }
  };

  useEffect(() => {
    const unlistenTranscription = listen<TranscriptionSegment>(
      "transcription-segment",
      (event) => {
        const segment = event.payload;

        console.log("[App] Received transcription segment:", segment);
        upsertTranscriptionSegment(segment);
      },
    );

    const unlistenVoiceActivity = listen<VoiceActivityEvent>(
      "voice-activity",
      (event) => {
        const { source, isActive, sessionId } = event.payload;
        console.log("Voice activity event:", event.payload);
        setVoiceActivity((prev) => {
          return {
            ...prev,
            [source]: {
              isActive,
              sessionId,
            },
          };
        });
      },
    );

    const unlistenBackendError = listen<BackendErrorEvent>(
      "backend-error",
      (event) => {
        const payload = event.payload;
        if (payload.message) {
          setError(payload.message);
        }
        if (payload.fallbackMode === "local") {
          setTranscriptionMode("local");
        }
      },
    );

    loadSettingsFromLocalStorage();
    refreshAllModels();
    loadLanguages();
    loadAudioDevices();
    checkMicPermission();
    loadStreamingConfig();
    loadWhisperParams();
    loadRecordingSaveConfig();
    loadScreenRecordingConfig();
    loadTranscriptionBackendConfig();

    return () => {
      unlistenTranscription.then((fn) => fn());
      unlistenVoiceActivity.then((fn) => fn());
      unlistenBackendError.then((fn) => fn());
    };
  }, [loadStreamingConfig, loadWhisperParams, loadTranscriptionBackendConfig, upsertTranscriptionSegment]);

  const getTotalMessageCount = useCallback(
    (sessions: SessionTranscription[]) =>
      sessions.reduce((total, session) => total + session.messages.length, 0),
    [],
  );

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  }, []);

  const handleScrollContainer = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) {
      return;
    }

    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const shouldPauseAutoScroll = distanceFromBottom > container.clientHeight;

    setIsAutoScrollPaused(shouldPauseAutoScroll);

    if (!shouldPauseAutoScroll) {
      setNewMessageCount(0);
    }
  }, []);

  useEffect(() => {
    const container = scrollContainerRef.current;
    const currentHeight = container?.scrollHeight ?? 0;
    const previousHeight = previousScrollHeightRef.current;
    const grewInHeight = currentHeight > previousHeight;
    previousScrollHeightRef.current = currentHeight;

    const currentCount = getTotalMessageCount(transcriptions);
    const previousCount = previousMessageCountRef.current;
    const incomingCount = Math.max(0, currentCount - previousCount);
    previousMessageCountRef.current = currentCount;

    if (incomingCount === 0 && !grewInHeight) {
      return;
    }

    if (isAutoScrollPaused) {
      setNewMessageCount((prev) => prev + (incomingCount > 0 ? incomingCount : 1));
      return;
    }

    scrollToBottom("smooth");
  }, [
    getTotalMessageCount,
    isAutoScrollPaused,
    scrollToBottom,
    transcriptions,
  ]);

  useEffect(() => {
    if (selectedModel) {
      localStorage.setItem("selectedModelPath", selectedModel);
    } else {
      localStorage.removeItem("selectedModelPath");
    }
  }, [selectedModel]);

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
        "check_microphone_permission",
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

  const handleTranscriptionModeChange = async (nextMode: "local" | "api") => {
    if (nextMode === transcriptionMode) {
      return;
    }

    await saveTranscriptionBackendConfig({
      mode: nextMode,
    });
  };

  const summarizeCurrentSession = async () => {
    if (isSummarizing || transcriptionMode !== "api") {
      return;
    }

    const latestSession =
      transcriptions.length > 0
        ? transcriptions[transcriptions.length - 1]
        : undefined;
    if (!latestSession || latestSession.messages.length === 0) {
      setError("要約対象のセッションがありません");
      return;
    }

    const text = latestSession.messages
      .map((m: TranscriptionSegment) => m.text)
      .join("\n")
      .trim();
    if (!text) {
      setError("要約対象のテキストがありません");
      return;
    }

    setIsSummarizing(true);
    try {
      const baseUrl = API_BASE_URL.replace(/\/$/, "");
      const response = await fetch(`${baseUrl}/api/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          session_id: latestSession.sessionId,
          source: latestSession.source,
        }),
      });

      if (!response.ok) {
        throw new Error(`Summary request failed: ${response.status}`);
      }

      const payload = (await response.json()) as SummaryResponse;
      const summary = payload.summary ?? payload.text ?? payload.result ?? "";
      if (!summary.trim()) {
        throw new Error("Summary response is empty");
      }

      setSummaryPanelText(summary.trim());
      setIsSummaryExpanded(true);
      setError("");
    } catch (err) {
      console.error("Failed to summarize session:", err);
      setError(
        `要約エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsSummarizing(false);
    }
  };

  const refreshAllModels = async () => {
    await scanForModels();
    await loadRemoteModels();
  };

  const handleLanguageChange = async (language: string) => {
    setSelectedLanguage(language);
    localStorage.setItem("selectedLanguage", language);

    try {
      await invoke("update_language", {
        language: language === "auto" ? null : language,
      });
    } catch (err) {
      console.error("Failed to update language:", err);
      setError(
        `言語変更エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
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
        }`,
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
        `モデル削除エラー: ${err instanceof Error ? err.message : String(err)}`,
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
    if (isMicBusy) {
      return;
    }

    if (isMuted) {
      await startMic();
    } else {
      await stopMic();
    }
  };

  const startMic = async () => {
    if (isMicBusy) {
      return;
    }

    setIsMicBusy(true);

    if (transcriptionMode === "local" && !isInitialized) {
      setError("モデルを初期化中です...");
      setIsMicBusy(false);
      return;
    }

    try {
      await invoke("start_mic", {
        language: selectedLanguage === "auto" ? null : selectedLanguage,
      });
      setIsMuted(false);
      setError("");
    } catch (err) {
      console.error("Mic start error:", err);
      setError(
        `マイク開始エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsMicBusy(false);
    }
  };

  const stopMic = async () => {
    if (isMicBusy) {
      return;
    }

    setIsMicBusy(true);

    setIsMuted(true);

    try {
      await invoke("stop_mic");
    } catch (err) {
      console.error("Mic stop error:", err);
      setError(
        `録音停止エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsMicBusy(false);
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
    setPlayingSessionKey(null);

    if (transcriptionSuppressedForPlaybackRef.current) {
      invoke("set_transcription_suppressed", { enabled: false })
        .then(() => {
          transcriptionSuppressedForPlaybackRef.current = false;
        })
        .catch((err) => {
          console.error("Failed to disable transcription suppression:", err);
          setError(
            `文字起こし再開エラー: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
        });
    }
  };

  const clearAllMessages = () => {
    stopAudio();
    setTranscriptions([]);
  };

  const handleCopySessionText = async (
    sessionKey: string,
    sessionText: string,
  ) => {
    try {
      await navigator.clipboard.writeText(sessionText);
      setCopiedSessionKey(sessionKey);
      if (copyFeedbackTimeoutRef.current) {
        window.clearTimeout(copyFeedbackTimeoutRef.current);
      }
      copyFeedbackTimeoutRef.current = window.setTimeout(() => {
        setCopiedSessionKey(null);
        copyFeedbackTimeoutRef.current = null;
      }, 1500);
    } catch (err) {
      setError(
        `コピーエラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  const formatSegmentTimestamp = (timestamp: number) => {
    const millis = timestamp < 1_000_000_000_000 ? timestamp * 1000 : timestamp;
    return new Date(millis).toLocaleString("ja-JP", {
      hour12: false,
    });
  };

  const handleCopyAllHistory = async () => {
    if (transcriptions.length === 0) {
      return;
    }

    const allMessages = transcriptions
      .flatMap((session) =>
        session.messages.map((message) => ({
          timestamp: message.timestamp,
          sender: message.source,
          text: message.text,
          sessionId: message.sessionId,
          messageId: message.messageId,
        })),
      )
      .sort((a, b) => {
        if (a.timestamp !== b.timestamp) {
          return a.timestamp - b.timestamp;
        }
        if (a.sessionId !== b.sessionId) {
          return a.sessionId - b.sessionId;
        }
        return a.messageId - b.messageId;
      });

    const exportText = allMessages
      .map(
        (message) =>
          `[${formatSegmentTimestamp(message.timestamp)}] ${message.sender}: ${message.text}`,
      )
      .join("\n");

    try {
      await navigator.clipboard.writeText(exportText);
      setCopiedAllHistory(true);
      if (copyAllFeedbackTimeoutRef.current) {
        window.clearTimeout(copyAllFeedbackTimeoutRef.current);
      }
      copyAllFeedbackTimeoutRef.current = window.setTimeout(() => {
        setCopiedAllHistory(false);
        copyAllFeedbackTimeoutRef.current = null;
      }, 1500);
    } catch (err) {
      setError(
        `履歴コピーエラー: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  useEffect(() => {
    return () => {
      if (copyFeedbackTimeoutRef.current) {
        window.clearTimeout(copyFeedbackTimeoutRef.current);
      }
      if (copyAllFeedbackTimeoutRef.current) {
        window.clearTimeout(copyAllFeedbackTimeoutRef.current);
      }
    };
  }, []);

  const playSessionAudio = async (audioData: number[], sessionKey: string) => {
    if (audioData.length === 0) {
      return;
    }
    try {
      if (playingSessionKey === sessionKey) {
        stopAudio();
        return;
      }

      if (currentAudioSource || currentAudioContext) {
        stopAudio();
      }

      if (!transcriptionSuppressedForPlaybackRef.current) {
        try {
          await invoke("set_transcription_suppressed", { enabled: true });
          transcriptionSuppressedForPlaybackRef.current = true;
        } catch (err) {
          console.error("Failed to enable transcription suppression:", err);
          setError(
            `文字起こし停止エラー: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
        }
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
      setPlayingSessionKey(sessionKey);

      source.onended = () => {
        audioContext.close();
        setCurrentAudioSource(null);
        setCurrentAudioContext(null);
        setPlayingSessionKey(null);
        if (transcriptionSuppressedForPlaybackRef.current) {
          invoke("set_transcription_suppressed", { enabled: false })
            .then(() => {
              transcriptionSuppressedForPlaybackRef.current = false;
            })
            .catch((err) => {
              console.error(
                "Failed to disable transcription suppression:",
                err,
              );
              setError(
                `文字起こし再開エラー: ${
                  err instanceof Error ? err.message : String(err)
                }`,
              );
            });
        }
      };

      source.start(0);
    } catch (err) {
      console.error("Audio playback error:", err);
      setError(
        `音声再生エラー: ${err instanceof Error ? err.message : String(err)}`,
      );
      setPlayingSessionKey(null);
      if (transcriptionSuppressedForPlaybackRef.current) {
        invoke("set_transcription_suppressed", { enabled: false })
          .then(() => {
            transcriptionSuppressedForPlaybackRef.current = false;
          })
          .catch((resumeErr) => {
            console.error(
              "Failed to disable transcription suppression:",
              resumeErr,
            );
          });
      }
    }
  };

  const isPendingVoiceSession = (source: VoiceSource) => {
    const state = voiceActivity[source];
    if (!state.isActive || state.sessionId === null) {
      return false;
    }
    if (
      !transcriptions.some(
        (session) =>
          session.source === source && session.sessionId === state.sessionId,
      )
    ) {
      return false;
    }
    return true;
  };

  const pendingUserSessionKey = isPendingVoiceSession("user");
  const pendingSystemSessionKey = isPendingVoiceSession("system");
  const showUserActivityIndicator =
    transcriptionMode !== "api" &&
    voiceActivity.user.isActive &&
    !pendingUserSessionKey;
  const showSystemActivityIndicator =
    transcriptionMode !== "api" &&
    voiceActivity.system.isActive &&
    !pendingSystemSessionKey;

  return (
    <div className="flex flex-col h-screen w-screen bg-base-100">
      {/* Header */}
      <header
        data-tauri-drag-region
        className="app-header bg-base-100 border-b border-base-200 flex items-center py-1 px-4 gap-4"
      >
        <div className="shrink-0 flex items-center gap-1">
          <button
            className="btn btn-ghost btn-sm btn-square"
            onClick={clearAllMessages}
            disabled={transcriptions.length === 0}
            title="メッセージをクリア"
            aria-label="メッセージをクリア"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            className="btn btn-ghost btn-sm btn-square"
            onClick={handleCopyAllHistory}
            disabled={transcriptions.length === 0}
            title={copiedAllHistory ? "履歴コピー完了" : "履歴をすべてコピー"}
            aria-label="履歴をすべてコピー"
          >
            {copiedAllHistory ? (
              <Check className="w-4 h-4" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>

        <div className="shrink-0 flex items-center gap-2">
          <input
            type="checkbox"
            className="toggle toggle-primary toggle-sm"
            checked={transcriptionMode === "api"}
            onChange={(e) =>
              void handleTranscriptionModeChange(
                e.target.checked ? "api" : "local",
              )
            }
            title="Pro モード切替"
          />
          <span className="text-xs font-semibold text-primary">Pro</span>
        </div>

        {transcriptionMode === "api" && (
          <button
            className={`btn btn-ghost btn-sm ${isSummarizing ? "btn-disabled" : ""}`}
            onClick={summarizeCurrentSession}
            disabled={isSummarizing || transcriptions.length === 0}
            title="今のセッションを要約"
          >
            {isSummarizing ? "要約中..." : "要約"}
          </button>
        )}

        <div className="flex-1"></div>

        <div className="flex items-center gap-2 shrink-0">
          <select
            value={selectedLanguage}
            onChange={(e) => handleLanguageChange(e.target.value)}
            className="select select-bordered select-xs w-24 font-normal"
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

          <div className="join">
            <button
              className={`join-item btn btn-sm ${
                transcriptionMode === "local" && !isInitialized
                  ? "btn-disabled"
                  : isMuted
                    ? "btn-ghost"
                    : "btn-primary"
              }`}
              onClick={toggleMute}
              disabled={
                isMicBusy || (transcriptionMode === "local" && !isInitialized)
              }
              title={
                isMicBusy ? "接続中..." : isMuted ? "マイクON" : "マイクOFF"
              }
            >
              {isMicBusy ? (
                <span className="loading loading-spinner loading-xs"></span>
              ) : isMuted ? (
                <MicOff className="w-4 h-4" />
              ) : (
                <Mic className="w-4 h-4" />
              )}
            </button>

            <button
              className={`join-item btn btn-sm ${
                isRecordingActive ? "btn-error" : "btn-ghost"
              } ${isRecordingBusy ? "btn-disabled" : ""}`}
              onClick={toggleRecording}
              disabled={isRecordingBusy}
              title={isRecordingActive ? "録画停止" : "録画開始"}
            >
              {isRecordingActive ? (
                <StopCircle className="w-4 h-4" />
              ) : (
                <Circle className="w-4 h-4" />
              )}
            </button>
          </div>

          <button
            className="btn btn-ghost btn-sm btn-square"
            onClick={() => setShowSettings(true)}
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative flex-1 overflow-hidden p-4">
        {summaryPanelText && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-3xl z-20">
            <div className="bg-base-200 border border-base-300 rounded-xl shadow-sm">
              <div className="px-3 py-2 flex items-center justify-between">
                <span className="text-sm font-medium">要約</span>
                <div className="flex items-center gap-1">
                  <button
                    className="btn btn-ghost btn-xs btn-square"
                    onClick={() => setIsSummaryExpanded((prev) => !prev)}
                    title={isSummaryExpanded ? "折りたたむ" : "展開する"}
                    aria-label={isSummaryExpanded ? "折りたたむ" : "展開する"}
                  >
                    {isSummaryExpanded ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                  <button
                    className="btn btn-ghost btn-xs btn-square"
                    onClick={() => setSummaryPanelText(null)}
                    title="閉じる"
                    aria-label="閉じる"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
              {isSummaryExpanded && (
                <div className="px-3 pb-3 text-sm whitespace-pre-wrap leading-relaxed border-t border-base-300/70">
                  {summaryPanelText}
                </div>
              )}
            </div>
          </div>
        )}

        <div
          ref={scrollContainerRef}
          onScroll={handleScrollContainer}
          className="h-full overflow-y-auto"
        >
          <div
            className={`max-w-3xl mx-auto space-y-3 ${
              summaryPanelText ? (isSummaryExpanded ? "pt-32" : "pt-16") : ""
            }`}
          >
            {error && (
              <div className="alert alert-error">
                <span className="text-sm">{error}</span>
              </div>
            )}

            {transcriptions.length === 0 &&
            !voiceActivity.user.isActive &&
            !voiceActivity.system.isActive ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center text-base-content/30 min-h-[50vh]">
                <MessageSquare className="w-16 h-16 mb-4 opacity-20" />
                <p className="text-sm font-medium">
                  マイクをオンにして、あなたの声を文字起こしします。
                </p>
                <p className="text-sm font-medium">
                  録音/録画をオンにして、システム音声を文字起こしします。
                </p>
              </div>
            ) : (
              <div className="flex flex-col">
                {transcriptions.map((session) => {
                  const alignment =
                    session.source === "user" ? "chat-end" : "chat-start";
                  const bubbleColor =
                    session.source === "user"
                      ? "chat-bubble-primary"
                      : "chat-bubble-secondary";
                  const sessionText = session.messages
                    .map((message) => message.text)
                    .join("\n");
                  const sessionAudio = session.messages
                    .map(
                      (message) => session.audioChunks[message.messageId] || [],
                    )
                    .flat();
                  return (
                    <div
                      key={session.sessionKey}
                      className={`chat ${alignment}`}
                    >
                      {session.messages.map((message, idx) => {
                        const messageKey = `${session.sessionKey}-${message.messageId}`;
                        return (
                          <div
                            key={messageKey}
                            className={`chat-bubble text-sm ${bubbleColor} ${
                              message.isFinal ? "" : "opacity-70"
                            } ${idx > 0 ? "mt-2" : ""}`}
                          >
                            <span className="flex-1 text-left">
                              {message.text}
                              {!message.isFinal && (
                                <span className="loading loading-dots loading-xs ml-1 align-bottom"></span>
                              )}
                            </span>
                          </div>
                        );
                      })}
                      <div className="chat-footer opacity-50 flex justify-between items-center">
                        <button
                          onClick={() =>
                            handleCopySessionText(
                              session.sessionKey,
                              sessionText,
                            )
                          }
                          className="btn btn-ghost btn-xs btn-circle"
                          title={
                            copiedSessionKey === session.sessionKey
                              ? "コピー完了"
                              : "コピー"
                          }
                        >
                          {copiedSessionKey === session.sessionKey ? (
                            <Check className="w-3 h-3" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                        {sessionAudio.length > 0 && (
                          <button
                            onClick={() =>
                              playSessionAudio(sessionAudio, session.sessionKey)
                            }
                            className="btn btn-ghost btn-xs btn-circle"
                          >
                            {playingSessionKey === session.sessionKey ? (
                              <Square className="w-3 h-3" />
                            ) : (
                              <Play className="w-3 h-3" />
                            )}
                          </button>
                        )}

                        <time className="text-[10px] opacity-60">
                          {new Date(
                            session.messages[0].timestamp,
                          ).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </time>
                      </div>
                    </div>
                  );
                })}
                {showUserActivityIndicator && (
                  <div className="chat chat-end">
                    <div className="chat-bubble chat-bubble-primary opacity-70 text-sm">
                      <span className="loading loading-dots loading-xs"></span>
                    </div>
                  </div>
                )}
                {showSystemActivityIndicator && (
                  <div className="chat chat-start">
                    <div className="chat-bubble chat-bubble-secondary opacity-70 text-sm">
                      <span className="loading loading-dots loading-xs"></span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {newMessageCount > 0 && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30">
            <button
              className="btn btn-ghost btn-sm border border-base-300 bg-base-100/95 text-base-content shadow-md"
              onClick={() => {
                scrollToBottom("smooth");
                setNewMessageCount(0);
                setIsAutoScrollPaused(false);
              }}
            >
              新規メッセージ {newMessageCount}件
            </button>
          </div>
        )}
      </main>

      {/* Settings Modal */}
      {showSettings && (
        <dialog className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-4 flex items-center justify-between">
              <span>設定</span>
              <label className="swap swap-rotate btn btn-ghost btn-circle btn-sm">
                <input
                  type="checkbox"
                  checked={theme === "dark"}
                  onChange={(e) =>
                    setTheme(e.target.checked ? "dark" : "light")
                  }
                />
                <Sun className="swap-off w-4 h-4" />
                <Moon className="swap-on w-4 h-4" />
              </label>
            </h3>

            <div className="space-y-3">
              <details
                className="collapse collapse-arrow bg-base-200/50 border border-base-300"
                open
              >
                <summary className="collapse-title text-sm font-semibold">
                  モデル設定
                </summary>
                <div className="collapse-content space-y-4">
                  <div className="form-control">
                    <label className="label">
                      <span className="label-text">使用モデル</span>
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
                                <p className="font-medium text-sm">
                                  {model.name}
                                </p>
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
                              <div className="flex flex-row flex-wrap gap-2 items-center justify-end">
                                {model.installed ? (
                                  <>
                                    <button
                                      className="btn btn-xs btn-outline text-[11px]"
                                      disabled={
                                        selectedModel === model.path ||
                                        !model.path ||
                                        modelOperations[model.id]
                                      }
                                      onClick={() =>
                                        model.path &&
                                        setSelectedModel(model.path)
                                      }
                                    >
                                      {selectedModel === model.path
                                        ? "使用中"
                                        : "使用する"}
                                    </button>
                                    <button
                                      className="btn btn-xs btn-error text-[11px]"
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
                                    className="btn btn-xs btn-primary text-[11px]"
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
                </div>
              </details>

              <details className="collapse collapse-arrow bg-base-200/50 border border-base-300">
                <summary className="collapse-title text-sm font-semibold">
                  マイク設定
                </summary>
                <div className="collapse-content">
                  <div className="space-y-4">
                    <div className="form-control">
                      <label className="label">
                        <span className="label-text">入力デバイス</span>
                      </label>
                      <select
                        value={selectedAudioDevice}
                        onChange={(e) =>
                          handleAudioDeviceChange(e.target.value)
                        }
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
                  </div>
                </div>
              </details>

              <details className="collapse collapse-arrow bg-base-200/50 border border-base-300">
                <summary className="collapse-title text-sm font-semibold">
                  録画・保存設定
                </summary>
                <div className="collapse-content space-y-4">
                  <div className="form-control">
                    <label className="label cursor-pointer">
                      <span className="label-text">録画/音声を保存</span>
                      <input
                        type="checkbox"
                        className="toggle toggle-primary"
                        checked={recordingSaveEnabled}
                        onChange={(e) =>
                          saveRecordingSaveConfig(
                            e.target.checked,
                            recordingSavePath,
                          )
                        }
                      />
                    </label>
                  </div>
                  {recordingSaveEnabled && (
                    <div className="form-control">
                      <label className="label">
                        <span className="label-text">保存先フォルダ</span>
                      </label>
                      <input
                        type="text"
                        placeholder="/path/to/save/folder"
                        value={recordingSavePath}
                        onChange={(e) => setRecordingSavePath(e.target.value)}
                        onBlur={() =>
                          saveRecordingSaveConfig(
                            recordingSaveEnabled,
                            recordingSavePath,
                          )
                        }
                        className="input input-bordered w-full"
                      />
                      <label className="label">
                        <span className="label-text-alt opacity-70">
                          録画時はMP4、音声のみの場合はWAVファイルとして保存されます
                        </span>
                      </label>
                    </div>
                  )}

                  <div className="form-control">
                    <label className="label cursor-pointer">
                      <span className="label-text">画面録画を有効化</span>
                      <input
                        type="checkbox"
                        className="toggle toggle-primary"
                        checked={screenRecordingEnabled}
                        onChange={(e) =>
                          saveScreenRecordingConfig(e.target.checked)
                        }
                      />
                    </label>
                    <label className="label">
                      <span className="label-text-alt opacity-70">
                        有効時は録画ボタンで画面+音声を録画、無効時は音声のみ保存
                      </span>
                    </label>
                  </div>
                </div>
              </details>

              <details className="collapse collapse-arrow bg-base-200/50 border border-base-300">
                <summary className="collapse-title text-sm font-semibold">
                  Whisper モデル設定
                </summary>
                <div className="collapse-content space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="label-text font-semibold">
                      Whisper モデル設定
                    </span>
                    <button
                      className={`btn btn-xs ${
                        isSavingWhisperParams ? "btn-disabled" : "btn-primary"
                      }`}
                      onClick={() => saveWhisperParams(whisperParams)}
                      disabled={isSavingWhisperParams}
                    >
                      {isSavingWhisperParams ? "保存中..." : "保存"}
                    </button>
                  </div>

                  <div className="space-y-2">
                    <label className="label">
                      <span className="label-text">
                        コンテキスト長 (audio_ctx: {whisperParams.audioCtx})
                      </span>
                    </label>
                    <p className="text-xs opacity-60">
                      長くするほど過去の音声を参照できますが、計算量とメモリ使用量が増えます。
                    </p>
                    <input
                      type="range"
                      min="50"
                      max="1500"
                      step="50"
                      value={whisperParams.audioCtx}
                      onChange={(e) =>
                        setWhisperParams((prev) => ({
                          ...prev,
                          audioCtx:
                            parseInt(e.target.value, 10) || prev.audioCtx,
                        }))
                      }
                      className="range range-sm range-primary"
                    />
                    <input
                      type="number"
                      min="50"
                      max="1500"
                      step="50"
                      value={whisperParams.audioCtx}
                      onChange={(e) =>
                        setWhisperParams((prev) => ({
                          ...prev,
                          audioCtx:
                            parseInt(e.target.value, 10) || prev.audioCtx,
                        }))
                      }
                      className="input input-bordered input-sm w-32"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="label">
                      <span className="label-text">
                        温度 (temperature:{" "}
                        {whisperParams.temperature.toFixed(2)})
                      </span>
                    </label>
                    <p className="text-xs opacity-60">
                      数値を上げると出力が多様になります。0に近いほど安定した結果になります。
                    </p>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={whisperParams.temperature}
                      onChange={(e) =>
                        setWhisperParams((prev) => ({
                          ...prev,
                          temperature:
                            parseFloat(e.target.value) ?? prev.temperature,
                        }))
                      }
                      className="range range-sm range-primary"
                    />
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      value={whisperParams.temperature}
                      onChange={(e) =>
                        setWhisperParams((prev) => ({
                          ...prev,
                          temperature:
                            parseFloat(e.target.value) ?? prev.temperature,
                        }))
                      }
                      className="input input-bordered input-sm w-32"
                    />
                  </div>
                </div>
              </details>

              <details className="collapse collapse-arrow bg-base-200/50 border border-base-300">
                <summary className="collapse-title text-sm font-semibold">
                  ストリーミング設定
                </summary>
                <div className="collapse-content space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="label-text font-semibold">
                      ストリーミング設定
                    </span>
                    <button
                      className={`btn btn-xs ${
                        isSavingStreamingConfig ? "btn-disabled" : "btn-primary"
                      }`}
                      onClick={() => saveStreamingConfig(streamingConfig)}
                      disabled={isSavingStreamingConfig}
                    >
                      {isSavingStreamingConfig ? "保存中..." : "保存"}
                    </button>
                  </div>

                  <div className="space-y-2">
                    <label className="label">
                      <span className="label-text">
                        VAD 閾値 ({streamingConfig.vadThreshold.toFixed(3)})
                      </span>
                    </label>
                    <p className="text-xs opacity-60">
                      数値が低いほど小さな声でも検知しやすく、高いほど大きな音しか検知しなくなります。
                    </p>
                    <input
                      type="range"
                      min="0.01"
                      max="0.99"
                      step="0.01"
                      value={streamingConfig.vadThreshold}
                      onChange={(e) =>
                        setStreamingConfig((prev) => ({
                          ...prev,
                          vadThreshold: parseFloat(e.target.value),
                        }))
                      }
                      className="range range-sm range-primary"
                    />
                    <input
                      type="number"
                      min="0.01"
                      max="0.99"
                      step="0.01"
                      value={streamingConfig.vadThreshold}
                      onChange={(e) =>
                        setStreamingConfig((prev) => ({
                          ...prev,
                          vadThreshold: parseFloat(e.target.value) || 0.1,
                        }))
                      }
                      className="input input-bordered input-sm w-32"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="label">
                      <span className="label-text">文字起こし間隔 (秒)</span>
                    </label>
                    <p className="text-xs opacity-60">
                      短くすると小刻みに更新され、長くするとまとまった文章で届きます。
                    </p>
                    <input
                      type="range"
                      min="0.5"
                      max="30"
                      step="0.5"
                      value={streamingConfig.partialIntervalSeconds}
                      onChange={(e) =>
                        setStreamingConfig((prev) => ({
                          ...prev,
                          partialIntervalSeconds: parseFloat(e.target.value),
                        }))
                      }
                      className="range range-sm range-primary"
                    />
                    <input
                      type="number"
                      min="0.5"
                      max="30"
                      step="0.5"
                      value={streamingConfig.partialIntervalSeconds}
                      onChange={(e) =>
                        setStreamingConfig((prev) => ({
                          ...prev,
                          partialIntervalSeconds:
                            parseFloat(e.target.value) || 4,
                        }))
                      }
                      className="input input-bordered input-sm w-32"
                    />
                  </div>
                </div>
              </details>

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
