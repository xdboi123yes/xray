/**
 * @file ThemeToggle.tsx
 * @description Standalone light/dark theme toggle button bound to the global store.
 */

import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useStore } from '../../store/useStore';

export const ThemeToggle: React.FC = () => {
  const { t } = useTranslation();
  const theme = useStore((state) => state.theme);
  const toggleTheme = useStore((state) => state.toggleTheme);

  const isDark = theme === 'dark';

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? t('themeLight', 'Light Theme') : t('themeDark', 'Dark Theme')}
      className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/60 px-3 py-2 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-100/70 dark:hover:bg-slate-800/40 transition-colors"
    >
      {isDark ? <Moon className="w-4 h-4 text-teal-500" /> : <Sun className="w-4 h-4 text-amber-500" />}
      <span>{isDark ? t('themeDark', 'Dark Theme') : t('themeLight', 'Light Theme')}</span>
    </button>
  );
};
