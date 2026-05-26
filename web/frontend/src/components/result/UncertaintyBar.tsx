/**
 * @file UncertaintyBar.tsx
 * @description Visualises Monte Carlo dropout variance with a clinical risk tier.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';

interface UncertaintyBarProps {
  mcVariance: number | null | undefined;
  mcPasses?: number | null;
}

export const UncertaintyBar: React.FC<UncertaintyBarProps> = ({ mcVariance, mcPasses }) => {
  const { t } = useTranslation();
  if (mcVariance === null || mcVariance === undefined) {
    return null;
  }
  const variance = Math.max(0, Math.min(0.25, mcVariance));
  const pct = (variance / 0.25) * 100;
  const tier = mcVariance > 0.12 ? 'high' : mcVariance > 0.06 ? 'medium' : 'low';
  const palette =
    tier === 'high'
      ? 'bg-rose-500'
      : tier === 'medium'
        ? 'bg-amber-500'
        : 'bg-teal-500';
  const tierLabel = t(`uncertaintyTier_${tier}`, tier.toUpperCase());

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 uppercase tracking-widest">
        <span>{t('labelUncertainty', 'Model Uncertainty (MC Variance)')}</span>
        <span>
          {mcVariance.toFixed(4)}
          {mcPasses ? ` · T=${mcPasses}` : ''} · {tierLabel}
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div className={`h-full ${palette} transition-all duration-500 rounded-full`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};
