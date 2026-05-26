/**
 * @file FlaggedBanner.tsx
 * @description Highlights samples flagged for clinician oversight due to high uncertainty.
 */

import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface FlaggedBannerProps {
  visible: boolean;
}

export const FlaggedBanner: React.FC<FlaggedBannerProps> = ({ visible }) => {
  const { t } = useTranslation();
  if (!visible) return null;

  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-2xl border border-rose-500/30 bg-rose-500/5 p-4 text-rose-700 dark:text-rose-300 text-sm"
    >
      <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
      <div>
        <p className="font-bold uppercase tracking-wide text-xs">
          {t('flaggedTitle', 'Flagged for Specialist Review')}
        </p>
        <p className="mt-1 text-xs leading-relaxed font-medium">
          {t(
            'flaggedBody',
            'This case shows high model uncertainty. Manual radiologist review is strongly recommended before any clinical decision.',
          )}
        </p>
      </div>
    </div>
  );
};
