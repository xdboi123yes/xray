/**
 * @file ConfidenceBar.tsx
 * @description Horizontal confidence visualization with a percentage label.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';

interface ConfidenceBarProps {
  value: number; // 0..1
  label?: string;
  variant?: 'positive' | 'neutral';
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ value, label, variant = 'positive' }) => {
  const { t } = useTranslation();
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const barColor =
    variant === 'positive' ? 'bg-emerald-500' : 'bg-slate-500';

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-widest">
        <span>{label ?? t('labelPrediction', 'Confidence')}</span>
        <span>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-500 rounded-full`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};
