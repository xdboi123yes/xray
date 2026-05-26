/**
 * @file ConformalSet.tsx
 * @description Renders the conformal prediction set (one or two members) with coverage chip.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import { ShieldCheck } from 'lucide-react';

interface ConformalSetProps {
  members?: string[] | null;
  coverage?: number | null;
}

export const ConformalSet: React.FC<ConformalSetProps> = ({ members, coverage }) => {
  const { t } = useTranslation();
  if (!members || members.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
        {t('labelConformalSet', 'Conformal Prediction Set')}
      </span>
      <div className="flex flex-wrap items-center gap-2">
        {members.map((m) => (
          <span
            key={m}
            className="inline-flex items-center gap-1 rounded-lg bg-slate-100 dark:bg-slate-800/60 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700/40 px-2 py-1 text-xs font-bold"
          >
            {m}
          </span>
        ))}
        {coverage !== null && coverage !== undefined && (
          <span className="inline-flex items-center gap-1 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/30 px-2 py-1 text-[10px] font-extrabold uppercase tracking-wider">
            <ShieldCheck className="w-3 h-3" />
            {(coverage * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
};
