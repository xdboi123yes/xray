/**
 * @file Sidebar.tsx
 * @description Modular Sidebar navigation panel utilizing NavLink for route synchronization.
 */

import React from 'react';
import { NavLink } from 'react-router-dom';
import { Shield, BarChart3, History, Info, Sparkles, Moon, Sun } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useStore } from '../../store/useStore';

export const Sidebar: React.FC = () => {
  const { theme, toggleTheme } = useStore();
  const { t, i18n } = useTranslation();

  const currentLang = i18n.language;
  const toggleLanguage = () => {
    const nextLang = currentLang === 'en' ? 'tr' : 'en';
    i18n.changeLanguage(nextLang);
    localStorage.setItem('language', nextLang);
  };

  const navItems = [
    { path: '/', label: t('navInference'), icon: Shield },
    { path: '/dashboard', label: t('navDashboard'), icon: BarChart3 },
    { path: '/history', label: t('navHistory', 'Diagnostics History'), icon: History },
    { path: '/ablation', label: t('navAblation', 'Ablation Studies'), icon: BarChart3 },
    { path: '/about', label: t('navAbout', 'Clinical Mechanics'), icon: Info },
  ];

  return (
    <aside className="w-full md:w-64 bg-white/70 dark:bg-slate-900/50 backdrop-blur-xl border-b md:border-b-0 md:border-r border-slate-200/50 dark:border-slate-800/50 flex flex-col shrink-0 z-30 print:hidden font-semibold">
      
      {/* Sidebar Logo Header */}
      <div className="p-6 border-b border-slate-200/50 dark:border-slate-800/50 flex items-center gap-3">
        <div className="p-2.5 bg-gradient-to-tr from-teal-500 to-emerald-500 rounded-xl text-white shadow-md shadow-teal-500/10">
          <Shield className="w-6 h-6" />
        </div>
        <div>
          <h1 className="font-heading font-extrabold text-xl tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-slate-900 to-slate-700 dark:from-white dark:to-slate-350 select-none">
            {t('appTitle')}
          </h1>
          <span className="text-[9px] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-widest block -mt-0.5">
            Clinician Suite
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `w-full flex items-center gap-3 px-4 py-3 rounded-xl text-xs font-semibold tracking-wide transition-all ${
                  isActive
                    ? 'bg-teal-500 text-white shadow-md shadow-teal-500/15 scale-[1.01]'
                    : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100/50 dark:hover:bg-slate-800/20 hover:text-slate-900 dark:hover:text-slate-100'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer Controls */}
      <div className="p-4 border-t border-slate-200/50 dark:border-slate-800/50 space-y-3">
        
        {/* Active Status Tag */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-teal-500/5 dark:bg-teal-500/10 border border-teal-500/25">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-[10px] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider flex items-center gap-1">
            <Sparkles className="w-3 h-3 animate-pulse" />
            {t('systemActive')}
          </span>
        </div>

        {/* Theme Toggler */}
        <button
          type="button"
          onClick={toggleTheme}
          className="w-full flex items-center justify-between px-4 py-2.5 border border-slate-200 dark:border-slate-800 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-350 hover:bg-slate-100/50 dark:hover:bg-slate-900/50 transition-colors"
        >
          <span className="flex items-center gap-2">
            {theme === 'dark' ? <Moon className="w-4 h-4 text-teal-500" /> : <Sun className="w-4 h-4 text-amber-500" />}
            {theme === 'dark' ? t('themeDark') : t('themeLight')}
          </span>
          <span className="text-[9px] bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 uppercase">
            {t('change')}
          </span>
        </button>

        {/* Language Switcher */}
        <button
          type="button"
          onClick={toggleLanguage}
          className="w-full flex items-center justify-between px-4 py-2.5 border border-slate-200 dark:border-slate-800 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-350 hover:bg-slate-100/50 dark:hover:bg-slate-900/50 transition-colors"
        >
          <span className="flex items-center gap-2">
            <span className="w-4 text-center font-bold text-[10px] text-teal-600 dark:text-teal-400 border border-teal-500/25 px-0.5 rounded">
              {i18n.language.toUpperCase()}
            </span>
            <span>{i18n.language === 'en' ? t('langEn') : t('langTr')}</span>
          </span>
          <span className="text-[9px] bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 uppercase">
            {t('toggleLanguage')}
          </span>
        </button>
      </div>

    </aside>
  );
};
