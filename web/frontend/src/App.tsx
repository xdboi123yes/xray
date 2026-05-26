/**
 * @file App.tsx
 * @description Application shell and routing controller.
 * Establishes sidebar navigation, localized layouts, responsive glass panels, and global dark mode bindings.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Heart } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useStore } from './store/useStore';
import { Sidebar } from './components/layout/Sidebar';
import { InferencePage } from './pages/InferencePage';
import { DashboardPage } from './pages/DashboardPage';
import { HistoryPage } from './pages/HistoryPage';
import { AblationPage } from './pages/AblationPage';
import { AboutPage } from './pages/AboutPage';

const App: React.FC = () => {
  const { initTheme } = useStore();
  const { t } = useTranslation();

  // Initialize persistent theme on mount
  useEffect(() => {
    initTheme();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 dark:bg-slate-950 dark:text-slate-100 flex flex-col md:flex-row transition-colors duration-200">
      
      {/* Sidebar Navigation Panel */}
      <Sidebar />

      {/* Main Clinical Content Area */}
      <main className="flex-1 flex flex-col min-w-0 min-h-screen relative z-10 print:p-0">
        
        {/* Content Wrapper */}
        <div className="flex-1 p-6 md:p-8 max-w-6xl w-full mx-auto print:p-0">
          <Routes>
            <Route path="/" element={<InferencePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/ablation" element={<AblationPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </div>

        {/* Global Footer (hidden during printing) */}
        <footer className="py-6 border-t border-slate-200/40 dark:border-slate-900 px-8 text-center text-[10px] text-slate-400 dark:text-slate-650 flex flex-col md:flex-row items-center justify-between gap-2 max-w-6xl w-full mx-auto print:hidden">
          <div className="flex items-center gap-1 font-semibold">
            <span>{t('copyright')}</span>
          </div>
          <div className="flex items-center gap-1 font-medium">
            <span>Made with</span>
            <Heart className="w-3 h-3 text-red-500 fill-red-500" />
            <span>for clinical visual intelligence.</span>
          </div>
        </footer>
      </main>

    </div>
  );
};

export default App;
