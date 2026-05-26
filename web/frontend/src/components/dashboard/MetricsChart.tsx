/**
 * @file MetricsChart.tsx
 * @description Recharts-powered line chart of ablation metrics (AUC / Accuracy / ECE).
 */

import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface AblationRow {
  ablation_id: string;
  provenance?: string;
  metrics: {
    auc_roc: number | null;
    accuracy: number | null;
    ece: number | null;
  };
}

interface MetricsChartProps {
  rows: AblationRow[];
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ rows }) => {
  const { t } = useTranslation();
  const data = useMemo(
    () =>
      rows.map((r) => ({
        name: r.ablation_id,
        auc: r.metrics.auc_roc !== null ? r.metrics.auc_roc * 100 : null,
        accuracy: r.metrics.accuracy !== null ? r.metrics.accuracy * 100 : null,
        ece: r.metrics.ece !== null ? r.metrics.ece * 100 : null,
      })),
    [rows],
  );

  return (
    <div className="rounded-2xl border border-slate-200/60 dark:border-slate-800/60 bg-white/50 dark:bg-slate-900/30 p-4">
      <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
        {t('metricsChartTitle', 'Ablation metrics comparison')}
      </h3>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
            <XAxis dataKey="name" fontSize={10} stroke="currentColor" />
            <YAxis fontSize={10} stroke="currentColor" />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,23,42,0.95)',
                border: 'none',
                borderRadius: 8,
                color: 'white',
                fontSize: 12,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="auc" stroke="#0d9488" strokeWidth={2} dot connectNulls={false} />
            <Line type="monotone" dataKey="accuracy" stroke="#6366f1" strokeWidth={2} dot connectNulls={false} />
            <Line type="monotone" dataKey="ece" stroke="#f59e0b" strokeWidth={2} dot connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
