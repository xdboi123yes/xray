/**
 * @file TierBadge.tsx
 * @description Tag showing which tier produced the prediction.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import { Zap, Layers } from 'lucide-react';

interface TierBadgeProps {
  tierUsed: number;
}

export const TierBadge: React.FC<TierBadgeProps> = ({ tierUsed }) => {
  const { t } = useTranslation();
  const isTier1 = tierUsed === 1;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider ${
        isTier1
          ? 'bg-teal-500/10 text-teal-600 border-teal-500/30'
          : 'bg-indigo-500/10 text-indigo-600 border-indigo-500/30'
      }`}
    >
      {isTier1 ? <Zap className="w-3 h-3" /> : <Layers className="w-3 h-3" />}
      {isTier1 ? t('badgeTier1', 'Tier 1 (Fast)') : t('badgeTier2', 'Tier 2 (Deep)')}
    </span>
  );
};
