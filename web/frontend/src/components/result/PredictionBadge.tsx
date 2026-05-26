/**
 * @file PredictionBadge.tsx
 * @description Color-coded pill summarising the final classification decision.
 */

import React from 'react';
import { CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface PredictionBadgeProps {
  prediction: string;
  flaggedForReview?: boolean;
}

export const PredictionBadge: React.FC<PredictionBadgeProps> = ({ prediction, flaggedForReview }) => {
  const { t } = useTranslation();
  const isPositive = prediction.toLowerCase().includes('pneumothorax');
  const isOOD = prediction.toLowerCase().includes('ood') || prediction.toLowerCase().includes('uncertain');

  let label = isPositive
    ? t('badgePneumothorax', 'Pneumothorax Positive')
    : t('badgeNormal', 'Normal');
  if (isOOD) label = t('badgeOOD', 'Out-of-Distribution');

  const palette = isOOD
    ? 'bg-amber-500/10 text-amber-600 border-amber-500/30'
    : isPositive
      ? 'bg-rose-500/10 text-rose-600 border-rose-500/30'
      : 'bg-emerald-500/10 text-emerald-600 border-emerald-500/30';

  const Icon = isOOD ? HelpCircle : isPositive ? AlertOctagon : CheckCircle2;

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-2xl border px-4 py-2 text-sm font-bold uppercase tracking-wide ${palette}`}
      data-flagged={Boolean(flaggedForReview)}
    >
      <Icon className="w-4 h-4" />
      {label}
      {flaggedForReview && (
        <span className="text-[10px] font-extrabold px-1.5 py-0.5 rounded bg-rose-600/20 text-rose-700 dark:text-rose-300">
          {t('badgeFlagged', 'Flagged for review')}
        </span>
      )}
    </span>
  );
};
