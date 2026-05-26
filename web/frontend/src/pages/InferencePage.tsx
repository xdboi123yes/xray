/**
 * @file InferencePage.tsx
 * @description Patient diagnostic page coordinator. Orchestrates visual file upload,
 * real-time progressive WebSocket feedback, final Grad-CAM overlays, and persistent SQLite history grids.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useEffect, useState } from 'react';
import { Shield, Clock, Calendar, CheckCircle2, AlertTriangle, Trash2, HelpCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store/useStore';
import { UploadZone } from '../components/UploadZone';
import { ProgressSteps } from '../components/ProgressSteps';
import { ResultCard } from '../components/ResultCard';

export const InferencePage: React.FC = () => {
  const { t } = useTranslation();
  const {
    activePrediction,
    inferenceProgress,
    inferenceSteps,
    inferenceLogs,
    isAnalyzing,
    analysisError,
    recentHistory,
    isHistoryLoading,
    fetchHistory,
    startInference,
    resetInference,
    deleteHistoryRecord,
    sessionFiles,
    displayPastPrediction,
  } = useStore();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Load history records on page mount
  useEffect(() => {
    fetchHistory();
    // Cleanup any hanging active analyses if page reloads
    resetInference();
  }, []);

  // Sync selectedFile when activePrediction changes
  useEffect(() => {
    if (activePrediction) {
      const cachedFile = sessionFiles[activePrediction.request_id] || null;
      setSelectedFile(cachedFile);
    } else {
      setSelectedFile(null);
    }
  }, [activePrediction, sessionFiles]);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    startInference(file);
  };

  const handleReset = () => {
    setSelectedFile(null);
    resetInference();
    fetchHistory();
  };

  const handleDeleteRecord = async (requestId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(t('deleteConfirm'))) {
      await deleteHistoryRecord(requestId);
    }
  };

  return (
    <div className="space-y-10 animate-fade-in print:space-y-0">
      
      {/* Dynamic Header (hidden during printing) */}
      <div className="print:hidden">
        <h1 className="font-heading font-extrabold text-3xl text-slate-800 dark:text-slate-100 tracking-tight flex items-center gap-2">
          <Shield className="w-8 h-8 text-teal-500" />
          {t('appSubTitle')}
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {t('inferenceDesc')}
        </p>
      </div>

      {/* Primary Interaction Board */}
      <div className="min-h-[400px] flex flex-col justify-center print:min-h-0">
        
        {/* State A: Idle (Renders UploadZone) */}
        {!isAnalyzing && !activePrediction && !analysisError && (
          <div className="space-y-6">
            <UploadZone onFileSelected={handleFileSelected} />
          </div>
        )}

        {/* State B: Active Prediction Stream (Renders ProgressSteps) */}
        {isAnalyzing && (
          <div className="py-6">
            <ProgressSteps
              progress={inferenceProgress}
              steps={inferenceSteps}
              logs={inferenceLogs}
            />
          </div>
        )}

        {/* State C: Inference Failed (Renders clean error page) */}
        {analysisError && (
          <div className="w-full max-w-xl mx-auto glass-panel border border-red-200 dark:border-red-900/30 rounded-2xl p-8 text-center space-y-6 shadow-lg animate-fade-in">
            <div className="w-16 h-16 bg-red-100 dark:bg-red-950/20 text-red-600 dark:text-red-400 rounded-full flex items-center justify-center mx-auto">
              <AlertTriangle className="w-8 h-8" />
            </div>
            
            <div className="space-y-2">
              <h3 className="font-heading font-bold text-lg text-slate-800 dark:text-slate-100">
                {t('inferenceFailed')}
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                {analysisError}
              </p>
            </div>

            <button
              onClick={handleReset}
              className="py-2.5 px-6 bg-teal-500 hover:bg-teal-600 active:scale-95 text-white text-xs font-bold rounded-xl transition-all duration-200 shadow-md shadow-teal-500/10"
            >
              {t('retry')}
            </button>
          </div>
        )}

        {/* State D: Complete (Renders ResultCard) */}
        {activePrediction && (
          <ResultCard
            prediction={activePrediction}
            originalFile={selectedFile}
            onReset={handleReset}
          />
        )}
      </div>

      {/* Past SQLite History Table (Rendered only on Idle upload state and hidden during printing) */}
      {!isAnalyzing && !activePrediction && !analysisError && (
        <div className="space-y-4 print:hidden animate-fade-in">
          <div className="border-t border-slate-200 dark:border-slate-800/80 pt-8 flex items-center justify-between">
            <h3 className="font-heading font-bold text-lg text-slate-800 dark:text-slate-100 flex items-center gap-2">
              <Clock className="w-5 h-5 text-slate-400" />
              {t('recentHistoryTitle')}
            </h3>
            {isHistoryLoading && (
              <span className="text-xs text-slate-400 font-medium">{t('updating')}</span>
            )}
          </div>

          {recentHistory.length > 0 ? (
            <div className="border border-slate-200 dark:border-slate-800/80 rounded-2xl overflow-hidden shadow-sm bg-white/30 dark:bg-slate-900/10">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100/50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-800/80 text-slate-400 font-bold uppercase tracking-wider">
                      <th className="p-4">{t('historyDate')}</th>
                      <th className="p-4">UUID / Request ID</th>
                      <th className="p-4">{t('historyResult')}</th>
                      <th className="p-4">{t('historyConfidence')}</th>
                      <th className="p-4">{t('historyTier')}</th>
                      <th className="p-4 text-center">{t('uncertainty')}</th>
                      <th className="p-4 text-right"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-150 dark:divide-slate-800/60 text-slate-600 dark:text-slate-350">
                    {recentHistory.map((record) => {
                      const isPneumo = record.prediction === 'Pneumothorax';
                      const formattedDate = new Date(record.timestamp).toLocaleString(undefined, {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      });

                      return (
                        <tr 
                          key={record.id}
                          onClick={() => displayPastPrediction(record.request_id)}
                          className="hover:bg-slate-150/40 dark:hover:bg-slate-900/30 transition-colors cursor-pointer select-none"
                          title={t('clickToView')}
                        >
                          {/* Date */}
                          <td className="p-4 font-semibold whitespace-nowrap flex items-center gap-1.5 text-slate-800 dark:text-slate-250">
                            <Calendar className="w-3.5 h-3.5 text-slate-400" />
                            {formattedDate}
                          </td>
                          
                          {/* Request ID */}
                          <td className="p-4 font-mono text-[10px] text-slate-400">
                            {record.request_id.substring(0, 18)}...
                          </td>

                          {/* Classification result badge */}
                          <td className="p-4">
                            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-bold text-[10px] uppercase border ${
                              isPneumo
                                ? 'bg-red-500/5 border-red-500/20 text-red-600 dark:text-red-400'
                                : 'bg-emerald-500/5 border-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                            }`}>
                              {isPneumo ? (
                                <AlertTriangle className="w-3 h-3" />
                              ) : (
                                <CheckCircle2 className="w-3 h-3" />
                              )}
                              {isPneumo ? t('badgePneumothorax') : t('badgeNormal')}
                            </span>
                          </td>

                          {/* Confidence level */}
                          <td className="p-4 font-bold text-slate-800 dark:text-slate-250">
                            %{Math.round(record.confidence * 100)}
                          </td>

                          {/* Active Tier */}
                          <td className="p-4">
                            <span className="font-semibold text-slate-500 dark:text-slate-400">
                              {record.tier_used === 1 ? 'Tier 1' : 'Tier 2'}
                            </span>
                          </td>

                          {/* MC Uncertainty Indicator */}
                          <td className="p-4 text-center">
                            {record.mc_variance !== null ? (
                              <span className={`font-mono font-bold ${
                                record.mc_variance > 0.12 ? 'text-rose-500' : 'text-emerald-500'
                              }`}>
                                {record.mc_variance.toFixed(4)}
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>

                          {/* Action button */}
                          <td className="p-4 text-right">
                            <button
                              type="button"
                              onClick={(e) => handleDeleteRecord(record.request_id, e)}
                              className="p-1.5 text-slate-400 hover:text-red-500 dark:hover:text-red-400 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-900 transition-all active:scale-90"
                              title={t('deleteRecord')}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center p-12 border border-dashed border-slate-350/50 dark:border-slate-800/80 rounded-2xl text-center text-slate-400 dark:text-slate-550 bg-white/20 dark:bg-slate-900/5">
              <HelpCircle className="w-10 h-10 text-slate-300 dark:text-slate-800 mb-2" />
              <p className="text-xs font-semibold">{t('noHistory')}</p>
            </div>
          )}
        </div>
      )}

    </div>
  );
};
