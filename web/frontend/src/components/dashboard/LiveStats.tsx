/**
 * @file LiveStats.tsx
 * @description Periodically polls /api/v1/health and renders system/GPU status.
 */

import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Activity, Cpu, Sparkles, Clock } from 'lucide-react';

interface HealthPayload {
  status: string;
  gpu: boolean;
  models_loaded: string[];
  version: string;
  uptime_s: number;
}

interface LiveStatsProps {
  pollIntervalMs?: number;
}

export const LiveStats: React.FC<LiveStatsProps> = ({ pollIntervalMs = 5000 }) => {
  const { t } = useTranslation();
  const [health, setHealth] = useState<HealthPayload | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchHealth = async () => {
      try {
        const res = await fetch('/api/v1/health');
        if (res.ok && !cancelled) {
          setHealth(await res.json());
        }
      } catch {
        // swallow; banner stays empty
      }
    };
    fetchHealth();
    const id = window.setInterval(fetchHealth, pollIntervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollIntervalMs]);

  if (!health) {
    return (
      <div className="rounded-2xl border border-slate-200/60 dark:border-slate-800/60 bg-white/50 dark:bg-slate-900/30 p-4 text-xs text-slate-400">
        {t('liveStatsConnecting', 'Connecting to backend...')}
      </div>
    );
  }

  const uptimeMin = Math.floor(health.uptime_s / 60);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <Tile icon={Activity} label={t('statStatus', 'Status')} value={health.status} accent="emerald" />
      <Tile icon={Cpu} label={t('statGpu', 'GPU')} value={health.gpu ? 'On' : 'Off'} accent="teal" />
      <Tile
        icon={Sparkles}
        label={t('statActiveModels', 'Models loaded')}
        value={String(health.models_loaded.length)}
        accent="indigo"
      />
      <Tile icon={Clock} label={t('statUptime', 'Uptime (min)')} value={String(uptimeMin)} accent="amber" />
    </div>
  );
};

type Accent = 'emerald' | 'teal' | 'indigo' | 'amber';

const tone: Record<Accent, string> = {
  emerald: 'bg-emerald-500/10 text-emerald-600',
  teal: 'bg-teal-500/10 text-teal-600',
  indigo: 'bg-indigo-500/10 text-indigo-600',
  amber: 'bg-amber-500/10 text-amber-600',
};

interface TileProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  accent: Accent;
}

const Tile: React.FC<TileProps> = ({ icon: Icon, label, value, accent }) => (
  <div className="rounded-2xl border border-slate-200/60 dark:border-slate-800/60 bg-white/60 dark:bg-slate-900/40 p-3 flex items-center gap-3">
    <div className={`p-2 rounded-xl ${tone[accent]}`}>
      <Icon className="w-4 h-4" />
    </div>
    <div className="flex flex-col">
      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</span>
      <span className="text-sm font-extrabold text-slate-900 dark:text-white">{value}</span>
    </div>
  </div>
);
