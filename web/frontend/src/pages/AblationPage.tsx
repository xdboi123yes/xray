/**
 * @file AblationPage.tsx
 * @description Renders the thesis clinical ablation studies table (A1-A15).
 * Visualizes model architectures, threshold configurations, and performance gains.
 * Honest provenance: preliminary rows display '—' for null metrics.
 */

import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Layers, Zap, CheckCircle2, Activity } from 'lucide-react';

interface AblationItem {
  ablation_id: string;
  name: string;
  description: string;
  run_id?: string;
  provenance?: string;
  metrics: {
    auc_roc: number | null;
    accuracy: number | null;
    ece: number | null;
  };
  tier1: string;
  tier2: string;
  routing: string;
  uncertainty?: string;
}

// Format a fractional metric (0-1) as a percentage with one decimal, or em-dash when null
const formatPct = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return '—';
  return `${(v * 100).toFixed(1)}%`;
};

const formatFloat = (v: number | null | undefined, digits = 3): string => {
  if (v === null || v === undefined) return '—';
  return v.toFixed(digits);
};

// Compute headline KPIs from real (non-null) rows only; never hard-code values
const computeKpis = (rows: AblationItem[]) => {
  const realRows = rows.filter((r) => r.metrics.auc_roc !== null);
  const aucPeak = realRows.length
    ? Math.max(...realRows.map((r) => r.metrics.auc_roc as number))
    : null;
  const eceBest = realRows.length
    ? Math.min(...realRows.filter((r) => r.metrics.ece !== null).map((r) => r.metrics.ece as number))
    : null;
  return { aucPeak, eceBest, realCount: realRows.length, totalCount: rows.length };
};

export const AblationPage: React.FC = () => {
  const { t } = useTranslation();
  const [ablationData, setAblationData] = useState<AblationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load ablation metrics on mount
  useEffect(() => {
    const fetchAblation = async () => {
      try {
        setLoading(true);
        const res = await fetch('/api/v1/ablation');
        if (res.ok) {
          const data = await res.json();
          setAblationData(data);
        } else if (res.status === 503) {
          setErrorMsg(
            t(
              'ablationNotGenerated',
              'Ablation results have not been generated yet. Run scripts/build_ablation_json.py after training.',
            ),
          );
        } else {
          setErrorMsg(`Unexpected status: ${res.status}`);
        }
      } catch (err) {
        setErrorMsg(String(err));
      } finally {
        setLoading(false);
      }
    };
    fetchAblation();
  }, [t]);

  const kpis = computeKpis(ablationData);
  const hasPreliminary = ablationData.some((r) => r.provenance === 'preliminary_placeholder');

  return (
    <div className="space-y-8 animate-fade-in">

      {/* Page Header */}
      <div>
        <h2 className="font-heading font-extrabold text-3xl text-slate-900 dark:text-white tracking-tight flex items-center gap-3">
          <Layers className="w-8 h-8 text-teal-500" />
          {t('ablationPageTitle', 'Thesis Ablation Experiments (A1 - A15)')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          {t(
            'ablationPageSubtitle',
            'Quantitative ablations validating tiered routing, MC Dropout, TTA, and conformal prediction.',
          )}
        </p>
      </div>

      {/* Global Provenance Banner */}
      {hasPreliminary && (
        <div className="bg-amber-500/5 border border-amber-500/30 rounded-2xl p-4 text-xs text-amber-700 dark:text-amber-400 font-medium">
          <strong className="font-bold uppercase tracking-wider mr-2">
            {t('preliminaryBannerTitle', 'Preliminary data')}
          </strong>
          {t(
            'preliminaryBannerBody',
            'Some rows have no genuine MLflow metrics yet — they are shown without numbers. After running real ablations, regenerate ablation.json via scripts/build_ablation_json.py.',
          )}
        </div>
      )}

      {/* Analytics Card Group (derived from real rows only) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-6 rounded-3xl backdrop-blur-xl flex items-center gap-4">
          <div className="p-3 bg-teal-500/10 text-teal-500 rounded-2xl">
            <Zap className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">
              {t('kpiAucPeak', 'AUC peak (real runs only)')}
            </span>
            <span className="text-2xl font-extrabold text-slate-900 dark:text-white block mt-0.5">
              {formatPct(kpis.aucPeak)}
            </span>
          </div>
        </div>

        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-6 rounded-3xl backdrop-blur-xl flex items-center gap-4">
          <div className="p-3 bg-emerald-500/10 text-emerald-500 rounded-2xl">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">
              {t('kpiRunsLoaded', 'Genuine MLflow runs loaded')}
            </span>
            <span className="text-2xl font-extrabold text-slate-900 dark:text-white block mt-0.5">
              {kpis.realCount} / {kpis.totalCount}
            </span>
          </div>
        </div>

        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-6 rounded-3xl backdrop-blur-xl flex items-center gap-4">
          <div className="p-3 bg-purple-500/10 text-purple-500 rounded-2xl">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">
              {t('kpiBestEce', 'Best Expected Calibration Error')}
            </span>
            <span className="text-2xl font-extrabold text-slate-900 dark:text-white block mt-0.5">
              {formatFloat(kpis.eceBest)}
            </span>
          </div>
        </div>
      </div>

      {/* Ablation Table */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t('loading')}</span>
        </div>
      ) : errorMsg ? (
        <div className="bg-rose-500/5 border border-rose-500/30 rounded-2xl p-6 text-rose-700 dark:text-rose-400 text-sm">
          {errorMsg}
        </div>
      ) : (
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl overflow-hidden backdrop-blur-xl shadow-xl shadow-slate-100/10 dark:shadow-none">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200/50 dark:border-slate-800/50 bg-slate-50/50 dark:bg-slate-900/50 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  <th className="px-6 py-4">Ablation ID</th>
                  <th className="px-6 py-4">Experiment</th>
                  <th className="px-6 py-4">MLflow Run</th>
                  <th className="px-6 py-4">Description</th>
                  <th className="px-6 py-4">Tier 1</th>
                  <th className="px-6 py-4">Tier 2</th>
                  <th className="px-6 py-4">AUC-ROC</th>
                  <th className="px-6 py-4">Accuracy</th>
                  <th className="px-6 py-4">ECE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200/30 dark:divide-slate-800/30 text-xs font-semibold text-slate-600 dark:text-slate-300">
                {ablationData.map((item) => (
                  <tr key={item.ablation_id} className="hover:bg-slate-100/20 dark:hover:bg-slate-800/10 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-teal-650 dark:text-teal-400 font-extrabold font-mono">
                      <div className="flex items-center gap-2">
                        {item.ablation_id}
                        {item.provenance === 'preliminary_placeholder' && (
                          <span className="bg-amber-500/10 text-amber-500 text-[8px] font-extrabold uppercase px-1.5 py-0.5 rounded border border-amber-500/20 tracking-wider">
                            {t('badgePreliminary', 'Preliminary')}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 font-bold text-slate-900 dark:text-white whitespace-nowrap">
                      {item.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono text-[10px] text-slate-500 dark:text-slate-450">
                      {item.run_id ? (
                        <span title={item.run_id} className="bg-slate-100 dark:bg-slate-800/50 px-2 py-1 rounded border border-slate-200 dark:border-slate-700/50 font-bold">
                          {item.run_id.substring(0, 8)}
                        </span>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-slate-450 dark:text-slate-400 font-normal leading-relaxed min-w-[200px]">
                      {item.description}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono text-[10px] uppercase text-slate-450 dark:text-slate-450">
                      {item.tier1}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono text-[10px] uppercase text-slate-450 dark:text-slate-450">
                      {item.tier2}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono font-bold text-slate-900 dark:text-white">
                      {formatPct(item.metrics.auc_roc)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono font-bold text-slate-900 dark:text-white">
                      {formatPct(item.metrics.accuracy)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-mono font-bold text-teal-600 dark:text-teal-400">
                      {formatFloat(item.metrics.ece)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};
