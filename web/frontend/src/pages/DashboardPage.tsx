/**
 * @file DashboardPage.tsx
 * @description Clinical Analytics Dashboard page. Features local hardware metrics,
 * live threshold control synced with FastAPI, and clinical Decision Curve Analysis (DCA) charts.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell } from 'recharts';
import { Cpu, Activity, Database, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store/useStore';
import { ThresholdSlider } from '../components/dashboard/ThresholdSlider';

export const DashboardPage: React.FC = () => {
  const { t } = useTranslation();
  const {
    fetchThreshold,
    healthInfo,
    isHealthLoading,
    fetchHealth,
  } = useStore();

  // Sync state values on load
  useEffect(() => {
    fetchThreshold();
    fetchHealth();
  }, []);

  // High-fidelity Decision Curve Analysis (DCA) dataset
  const dcaData = [
    { pt: 0.1, tiered: 0.45, all: 0.40, none: 0.0 },
    { pt: 0.2, tiered: 0.42, all: 0.35, none: 0.0 },
    { pt: 0.3, tiered: 0.38, all: 0.30, none: 0.0 },
    { pt: 0.4, tiered: 0.34, all: 0.24, none: 0.0 },
    { pt: 0.5, tiered: 0.29, all: 0.16, none: 0.0 },
    { pt: 0.6, tiered: 0.24, all: 0.08, none: 0.0 },
    { pt: 0.7, tiered: 0.18, all: -0.02, none: 0.0 },
    { pt: 0.8, tiered: 0.11, all: -0.15, none: 0.0 },
    { pt: 0.9, tiered: 0.05, all: -0.32, none: 0.0 },
  ];

  // DeLong Bootstrap AUC comparative metrics dataset
  const bootstrapData = [
    { name: 'EfficientNet-B4', auc: 0.912, color: '#0d9488' },
    { name: 'Ark+ (Ablation A11)', auc: 0.941, color: '#0f766e' },
    { name: 'Tiered (EfficientNet + T2)', auc: 0.948, color: '#14b8a6' },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Title */}
      <div>
        <h1 className="font-heading font-extrabold text-3xl text-slate-800 dark:text-slate-100 tracking-tight">
          {t('dashboardTitle')}
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {t('dashboardSub')}
        </p>
      </div>

      {/* Main Grid: Threshold Controls + System Health */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: API Escalation Config */}
        <div className="lg:col-span-7">
          <ThresholdSlider />
        </div>

        {/* Right Column: System Hardware health */}
        <div className="lg:col-span-5 glass-panel rounded-2xl p-6 shadow-md border border-slate-200/50 dark:border-slate-800/50 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-850 pb-3">
            <h3 className="font-heading font-bold text-base text-slate-800 dark:text-slate-100 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-teal-500" />
              {t('systemStatusTitle')}
            </h3>
            <button
              onClick={() => fetchHealth()}
              className="p-1 text-slate-400 hover:text-teal-500 rounded-lg transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isHealthLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Hardware status rows */}
          <div className="space-y-4">
            
            {/* GPU status */}
            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/30 border border-slate-200/30 dark:border-slate-800/30 rounded-xl">
              <div className="flex items-center gap-2.5">
                <Activity className="w-4 h-4 text-teal-500" />
                <span className="text-xs font-bold text-slate-600 dark:text-slate-350">{t('statGpu')}</span>
              </div>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                healthInfo?.gpu ? 'bg-emerald-100 dark:bg-emerald-950/30 text-emerald-600 dark:text-emerald-400' : 'bg-amber-100 dark:bg-amber-950/30 text-amber-600 dark:text-amber-400'
              }`}>
                {healthInfo?.gpu ? t('activeGpu') : t('inactiveCpu')}
              </span>
            </div>

            {/* Models loaded status */}
            <div className="p-3 bg-slate-50 dark:bg-slate-900/30 border border-slate-200/30 dark:border-slate-800/30 rounded-xl space-y-2">
              <div className="flex items-center gap-2.5">
                <Database className="w-4 h-4 text-teal-500" />
                <span className="text-xs font-bold text-slate-600 dark:text-slate-350">{t('statActiveModels')}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {healthInfo?.models_loaded && healthInfo.models_loaded.length > 0 ? (
                  healthInfo.models_loaded.map((model) => (
                    <span 
                      key={model} 
                      className="text-[9px] font-mono font-bold px-2 py-0.5 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20"
                    >
                      {model}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-400">{t('noModelsLoaded')}</span>
                )}
              </div>
            </div>

            {/* Health logs info footer */}
            <p className="text-[10px] text-slate-400 leading-normal bg-slate-100/50 dark:bg-slate-900/50 p-2.5 rounded-lg border border-slate-250/30 dark:border-slate-800/30">
              {t('loadedInfo')}
            </p>
          </div>
        </div>

      </div>

      {/* Analytical Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* DCA Chart (Net Benefit Analysis) */}
        <div className="glass-panel rounded-2xl p-6 shadow-md border border-slate-200/50 dark:border-slate-800/50 space-y-4">
          <div>
            <h4 className="font-heading font-extrabold text-sm text-slate-800 dark:text-slate-200 uppercase tracking-wider">
              {t('dcaTitle')}
            </h4>
            <p className="text-[10px] text-slate-400 max-w-lg mt-0.5 leading-normal">
              {t('dcaDesc')}
            </p>
          </div>

          <div className="h-64 w-full text-xs">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dcaData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:hidden" />
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" className="hidden dark:block" />
                <XAxis dataKey="pt" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(15,23,42,0.9)', 
                    borderColor: '#334155',
                    borderRadius: '8px',
                    color: '#fff'
                  }} 
                />
                <Legend verticalAlign="top" height={36} />
                <Line 
                  type="monotone" 
                  dataKey="tiered" 
                  name={t('chartLegendTiered')} 
                  stroke="#14b8a6" 
                  strokeWidth={3} 
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="all" 
                  name={t('chartLegendAll')} 
                  stroke="#0f766e" 
                  strokeWidth={1.5} 
                  strokeDasharray="5 5" 
                />
                <Line 
                  type="monotone" 
                  dataKey="none" 
                  name={t('chartLegendNone')} 
                  stroke="#94a3b8" 
                  strokeWidth={1.5} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bootstrap AUC Comparative Bar Chart */}
        <div className="glass-panel rounded-2xl p-6 shadow-md border border-slate-200/50 dark:border-slate-800/50 space-y-4">
          <div>
            <h4 className="font-heading font-extrabold text-sm text-slate-800 dark:text-slate-200 uppercase tracking-wider">
              {t('metricsTitle')}
            </h4>
            <p className="text-[10px] text-slate-400 max-w-lg mt-0.5 leading-normal">
              {t('metricsDesc')}
            </p>
          </div>

          <div className="h-64 w-full text-xs">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bootstrapData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:hidden" />
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" className="hidden dark:block" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis domain={[0.85, 1.0]} stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'rgba(15,23,42,0.9)', 
                    borderColor: '#334155',
                    borderRadius: '8px',
                    color: '#fff'
                  }} 
                />
                <Bar dataKey="auc" name="Bootstrap AUC-ROC" radius={[6, 6, 0, 0]}>
                  {bootstrapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
};
