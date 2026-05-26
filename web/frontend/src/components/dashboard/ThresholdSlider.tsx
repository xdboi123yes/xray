/**
 * @file ThresholdSlider.tsx
 * @description Encapsulated Operating Threshold Slider.
 * Syncs value changes with debounced PUT requests to the FastAPI backend.
 */

import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sliders, RefreshCw, Check, AlertCircle } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const ThresholdSlider: React.FC = () => {
  const { t } = useTranslation();
  const {
    thresholdValue,
    thresholdMode,
    isThresholdLoading,
    thresholdError,
    fetchThreshold,
    updateThreshold,
  } = useStore();

  const [localVal, setLocalVal] = useState<number>(thresholdValue);
  const [successToast, setSuccessToast] = useState<boolean>(false);

  // Sync state values on load
  useEffect(() => {
    fetchThreshold();
  }, []);

  // Sync local slider when store value changes
  useEffect(() => {
    setLocalVal(thresholdValue);
  }, [thresholdValue]);

  // Debounced/Buffered threshold updates
  const handleThresholdChange = async (val: number) => {
    setLocalVal(val);
    const success = await updateThreshold(val, thresholdMode);
    if (success) {
      setSuccessToast(true);
      setTimeout(() => setSuccessToast(false), 3000);
    }
  };

  const handleModeChange = async (mode: 'static' | 'dynamic') => {
    const success = await updateThreshold(localVal, mode);
    if (success) {
      setSuccessToast(true);
      setTimeout(() => setSuccessToast(false), 3000);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-md border border-slate-200/50 dark:border-slate-800/50 space-y-6">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-850 pb-3">
        <h3 className="font-heading font-bold text-base text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <Sliders className="w-5 h-5 text-teal-500" />
          {t('thresholdControlTitle')}
        </h3>
        {isThresholdLoading && (
          <RefreshCw className="w-4 h-4 text-teal-500 animate-spin" />
        )}
      </div>

      <p className="text-xs text-slate-505 dark:text-slate-400 leading-relaxed font-semibold">
        {t('thresholdDesc')}
      </p>

      {/* Mode Selectors */}
      <div className="space-y-3">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">
          {t('thresholdMode')}
        </span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => handleModeChange('static')}
            className={`flex-1 py-2.5 px-4 rounded-xl font-semibold text-xs transition-all border ${
              thresholdMode === 'static'
                ? 'bg-teal-500 text-white border-teal-500 shadow-sm shadow-teal-500/10'
                : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
            }`}
          >
            {t('modeStatic')}
          </button>
          <button
            type="button"
            onClick={() => handleModeChange('dynamic')}
            className={`flex-1 py-2.5 px-4 rounded-xl font-semibold text-xs transition-all border ${
              thresholdMode === 'dynamic'
                ? 'bg-teal-500 text-white border-teal-500 shadow-sm shadow-teal-500/10'
                : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
            }`}
          >
            {t('modeDynamic')}
          </button>
        </div>
      </div>

      {/* Slider Range */}
      <div className="space-y-4">
        <div className="flex items-center justify-between font-semibold">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
            {t('thresholdLabel')}
          </span>
          <span className="font-heading font-extrabold text-xl text-teal-500">
            {localVal.toFixed(2)}
          </span>
        </div>

        <div className="space-y-2">
          <input
            type="range"
            min="0.50"
            max="0.95"
            step="0.01"
            value={localVal}
            onChange={(e) => setLocalVal(parseFloat(e.target.value))}
            onMouseUp={() => handleThresholdChange(localVal)}
            className="w-full h-2 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-bold uppercase tracking-wide">
            <span>{t('fastT1Decision')}</span>
            <span>{t('defaultVal')}</span>
            <span>{t('strictEscalation')}</span>
          </div>
        </div>
      </div>

      {/* Success / Error Messages */}
      {successToast && (
        <div className="flex items-center gap-2 p-3 bg-teal-50 dark:bg-teal-950/20 border border-teal-200 dark:border-teal-900/30 text-teal-600 dark:text-teal-400 rounded-xl text-xs font-semibold animate-fade-in">
          <Check className="w-4 h-4" />
          {t('thresholdSaveSuccess')}
        </div>
      )}

      {thresholdError && (
        <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 text-red-600 dark:text-red-400 rounded-xl text-xs font-semibold">
          <AlertCircle className="w-4 h-4" />
          {t('thresholdSaveError')}: {thresholdError}
        </div>
      )}
    </div>
  );
};
