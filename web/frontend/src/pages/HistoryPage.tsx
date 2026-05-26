/**
 * @file HistoryPage.tsx
 * @description Renders the SQLite-backed clinical diagnostic log history.
 * Implements search filters, detail overlays, and record removals.
 */

import React, { useEffect, useState } from 'react';
import { useTranslation as useI18n } from 'react-i18next';
import { Search, Trash2, Calendar, FileText, ChevronRight, X, AlertCircle } from 'lucide-react';

interface HistoryRecord {
  id: number;
  request_id: string;
  prediction: string;
  confidence: number;
  tier_used: number;
  mc_variance: number | null;
  flagged_for_review: boolean;
  timestamp: string;
}

export const HistoryPage: React.FC = () => {
  const { t } = useI18n();
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);

  // Load diagnostic logs on mount
  const fetchHistory = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/v1/history?limit=100');
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Handle record deletion
  const handleDelete = async (requestId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent opening modal
    if (!window.confirm(t('deleteConfirm'))) return;

    try {
      const res = await fetch(`/api/v1/history/${requestId}`, { method: 'DELETE' });
      if (res.ok) {
        setHistory(history.filter((rec) => rec.request_id !== requestId));
        if (selectedRecord?.request_id === requestId) {
          setSelectedRecord(null);
        }
      }
    } catch (err) {
      console.error('Error deleting record:', err);
    }
  };

  // Filter history based on search query
  const filteredHistory = history.filter((rec) => {
    const term = search.toLowerCase();
    return (
      rec.request_id.toLowerCase().includes(term) ||
      rec.prediction.toLowerCase().includes(term) ||
      (rec.tier_used === 1 ? 'tier 1' : 'tier 2').includes(term)
    );
  });

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Page Header */}
      <div>
        <h2 className="font-heading font-extrabold text-3xl text-slate-900 dark:text-white tracking-tight">
          {t('recentHistoryTitle')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Review recent diagnostic inferences stored locally in SQLite database.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex items-center gap-4 bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-4 rounded-2xl backdrop-blur-xl">
        <div className="relative flex-1">
          <Search className="absolute left-4 top-3.5 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search by UUID, prediction or classification tier..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-11 pr-4 py-3 bg-slate-100/50 dark:bg-slate-950/50 border border-slate-200/50 dark:border-slate-800/50 rounded-xl text-xs font-semibold placeholder-slate-400 focus:outline-none focus:border-teal-500/50 transition-colors"
          />
        </div>
      </div>

      {/* History Records List */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t('loading')}</span>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl p-12 text-center backdrop-blur-xl">
          <AlertCircle className="w-10 h-10 text-slate-400 mx-auto mb-3" />
          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{t('noHistory')}</p>
        </div>
      ) : (
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl overflow-hidden backdrop-blur-xl shadow-xl shadow-slate-100/10 dark:shadow-none">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200/50 dark:border-slate-800/50 bg-slate-50/50 dark:bg-slate-900/50 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  <th className="px-6 py-4">{t('historyDate')}</th>
                  <th className="px-6 py-4">Request UUID</th>
                  <th className="px-6 py-4">{t('historyResult')}</th>
                  <th className="px-6 py-4">{t('historyTier')}</th>
                  <th className="px-6 py-4">{t('historyConfidence')}</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200/30 dark:divide-slate-800/30 text-xs font-semibold text-slate-600 dark:text-slate-300">
                {filteredHistory.map((rec) => {
                  const isPneumo = rec.prediction === 'Pneumothorax';
                  const date = new Date(rec.timestamp).toLocaleString();
                  return (
                    <tr
                      key={rec.id}
                      onClick={() => setSelectedRecord(rec)}
                      className="hover:bg-slate-100/20 dark:hover:bg-slate-800/10 cursor-pointer transition-colors group"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-slate-400 dark:text-slate-500 font-medium">
                        <span className="flex items-center gap-2">
                          <Calendar className="w-3.5 h-3.5" />
                          {date}
                        </span>
                      </td>
                      <td className="px-6 py-4 font-mono text-[10px] tracking-tight whitespace-nowrap">
                        {rec.request_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider ${
                            isPneumo
                              ? 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
                              : 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                          }`}
                        >
                          {isPneumo ? t('badgePneumothorax') : t('badgeNormal')}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 text-[10px] rounded text-slate-400 font-bold uppercase">
                          Tier {rec.tier_used}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap font-mono text-slate-900 dark:text-white">
                        {(rec.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            type="button"
                            onClick={(e) => handleDelete(rec.request_id, e)}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/5 dark:hover:bg-rose-500/10 transition-colors"
                            title={t('deleteRecord')}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                          <ChevronRight className="w-4 h-4 text-slate-350 group-hover:translate-x-0.5 transition-transform" />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Diagnostic Record Details Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-md flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="w-full max-w-2xl bg-white dark:bg-slate-900 border border-slate-200/50 dark:border-slate-800/50 rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-200/50 dark:border-slate-800/50 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-teal-500/10 text-teal-500 rounded-xl">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white">
                    {t('resultCardTitle')}
                  </h3>
                  <span className="font-mono text-[9px] text-slate-450 tracking-tight block">
                    UUID: {selectedRecord.request_id}
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedRecord(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-6 flex-1 text-slate-700 dark:text-slate-350">
              
              {/* Decision Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1">
                    {t('labelPrediction')}
                  </span>
                  <span className={`text-sm font-bold block ${selectedRecord.prediction === 'Pneumothorax' ? 'text-rose-500' : 'text-emerald-500'}`}>
                    {selectedRecord.prediction === 'Pneumothorax' ? t('badgePneumothorax') : t('badgeNormal')}
                  </span>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1">
                    {t('labelConfidence')}
                  </span>
                  <span className="text-sm font-extrabold text-slate-900 dark:text-white block">
                    {(selectedRecord.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1">
                    {t('labelTierUsed')}
                  </span>
                  <span className="text-sm font-bold text-slate-900 dark:text-white block">
                    Tier {selectedRecord.tier_used} ({selectedRecord.tier_used === 1 ? 'MobileNetV3' : 'Escalated Specialist'})
                  </span>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1">
                    {t('labelUncertainty')}
                  </span>
                  <span className="text-sm font-bold text-slate-900 dark:text-white block font-mono">
                    {selectedRecord.mc_variance !== null ? selectedRecord.mc_variance.toFixed(4) : 'N/A (Bypassed)'}
                  </span>
                </div>
              </div>

              {/* Archive Disclaimer */}
              <div className="p-4 bg-amber-500/5 dark:bg-amber-500/10 border border-amber-500/20 rounded-2xl flex gap-3">
                <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-xs text-amber-600 dark:text-amber-400 uppercase tracking-wide">
                    {t('originalImageArchived')}
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">
                    {t('originalImageArchivedDesc')}
                  </p>
                </div>
              </div>

              {/* Clinical Disclaimer */}
              <div className="p-4 bg-slate-50 dark:bg-slate-950/50 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl space-y-1.5">
                <h4 className="font-bold text-[10px] text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5" />
                  {t('disclaimerTitle')}
                </h4>
                <p className="text-[10px] text-slate-400 dark:text-slate-500 leading-relaxed font-medium">
                  {t('disclaimerContent')}
                </p>
              </div>

            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-slate-200/50 dark:border-slate-800/50 bg-slate-50 dark:bg-slate-950/30 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={() => setSelectedRecord(null)}
                className="px-5 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 dark:bg-slate-850 dark:hover:bg-slate-800 dark:text-slate-200 rounded-xl text-xs font-bold transition-colors"
              >
                Close
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
};
