import { useState, useCallback, useMemo } from "react";
import type { Translations } from "./types";
import en from "./en";
import ja from "./ja";

export type Locale = "en" | "ja";

const translations: Record<Locale, Translations> = { en, ja };

function detectLocale(): Locale {
  const saved = localStorage.getItem("uiLocale");
  if (saved === "en" || saved === "ja") return saved;

  const lang = navigator.language.toLowerCase();
  if (lang.startsWith("ja")) return "ja";
  return "en";
}

export function useLocale() {
  const [locale, setLocaleState] = useState<Locale>(detectLocale);

  const setLocale = useCallback((next: Locale) => {
    localStorage.setItem("uiLocale", next);
    setLocaleState(next);
  }, []);

  const t = useMemo(() => translations[locale], [locale]);

  return { locale, setLocale, t };
}

export type { Translations };
