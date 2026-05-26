/**
 * @file ResultCard.tsx
 * @description Premium diagnostic outcome report component. Renders tabbed view,
 * interactive opacity overlay, uncertainty metrics, conformal boundaries, and print capabilities.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useState } from 'react';
import { Shield, Percent, FileText, AlertTriangle, HelpCircle, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { PredictionDTO } from '../store/useStore';
import { useStore } from '../store/useStore';
import { GradCAMViewer } from './gradcam/GradCAMViewer';
import { exportClinicalPdfReport } from '../utils/pdf-report';

// Decoupled presentation sub-components
import { FlaggedBanner } from './result/FlaggedBanner';
import { PredictionBadge } from './result/PredictionBadge';
import { TierBadge } from './result/TierBadge';
import { ConfidenceBar } from './result/ConfidenceBar';
import { UncertaintyBar } from './result/UncertaintyBar';
import { ConformalSet } from './result/ConformalSet';

interface ResultCardProps {
  prediction: PredictionDTO;
  originalFile?: File | null;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({ prediction, originalFile, onReset }) => {
  const { t, i18n } = useTranslation();
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const thresholdValue = useStore((state) => state.thresholdValue);

  // Helper utility to convert file to Base64 asynchronously
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
    });
  };

  // Generate local Object URL for the uploaded file to render in UI
  const originalUrl = originalFile ? URL.createObjectURL(originalFile) : null;

  // Formatted sources for the GradCAMViewer component
  const gradcamTier1Src = prediction.gradcam_tier1_b64
    ? (prediction.gradcam_tier1_b64.startsWith('data:') ? prediction.gradcam_tier1_b64 : `data:image/png;base64,${prediction.gradcam_tier1_b64}`)
    : undefined;

  const gradcamTier2Src = prediction.gradcam_tier2_b64
    ? (prediction.gradcam_tier2_b64.startsWith('data:') ? prediction.gradcam_tier2_b64 : `data:image/png;base64,${prediction.gradcam_tier2_b64}`)
    : undefined;

  const gradcamSrc = gradcamTier2Src || gradcamTier1Src;

  const isPneumothorax = prediction.prediction === 'Pneumothorax';
  const isOod = prediction.flagged_for_review && (prediction.mc_variance || 0) > 0.15; // Simulated OOD logic based on MC Variance

  // Trigger bilingual high-fidelity PDF report generation
  const handleExportPdf = async () => {
    setIsExporting(true);
    try {
      let originalB64: string | undefined = undefined;
      if (originalFile) {
        originalB64 = await fileToBase64(originalFile);
      }

      await exportClinicalPdfReport({
        requestId: prediction.request_id,
        prediction: prediction.prediction,
        confidence: prediction.confidence,
        tierUsed: prediction.tier_used,
        mcVariance: prediction.mc_variance,
        timestamp: prediction.timestamp,
        originalImageB64: originalB64,
        gradcamImageB64: gradcamSrc,
        conformalSet: prediction.conformal_set || undefined,
        activeThreshold: thresholdValue,
        language: i18n.language || 'en',
      });
    } catch (error) {
      console.error('Failed to export clinical PDF report:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fade-in print:p-0 print:shadow-none">
      
      {/* Clinician Oversight Warning Banner */}
      <FlaggedBanner visible={prediction.flagged_for_review} />

      {/* Main Diagnostic Card */}
      <div className="glass-panel rounded-2xl shadow-xl overflow-hidden border border-slate-200/50 dark:border-slate-800/50 print:border-none print:shadow-none">
        
        {/* Diagnostic Banner Color Code Header */}
        <div className={`p-6 border-b text-white flex flex-col md:flex-row md:items-center justify-between gap-4 print:bg-slate-100 print:text-slate-900 print:border-slate-300 ${
          isOod
            ? 'bg-gradient-to-r from-amber-500 to-orange-600'
            : isPneumothorax
            ? 'bg-gradient-to-r from-rose-500 to-red-600'
            : 'bg-gradient-to-r from-emerald-500 to-teal-600'
        }`}>
          <div>
            <div className="text-[10px] uppercase font-bold tracking-widest text-white/80 print:text-slate-500">
              {t('resultCardTitle')}
            </div>
            <div className="flex items-center gap-3 mt-1 flex-wrap">
              <h2 className="font-heading font-extrabold text-2xl print:text-slate-950">
                {isOod ? t('badgeOOD') : isPneumothorax ? t('badgePneumothorax') : t('badgeNormal')}
              </h2>
              <PredictionBadge prediction={prediction.prediction} flaggedForReview={prediction.flagged_for_review} />
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleExportPdf}
              disabled={isExporting}
              className="flex items-center gap-1.5 px-4 py-2 bg-white/20 hover:bg-white/30 text-white font-semibold text-xs rounded-xl transition-all duration-200 active:scale-95 print:hidden disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
              {isExporting ? t('exportingPdf') || 'Exporting...' : t('exportPdf')}
            </button>
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 px-4 py-2 bg-black/20 hover:bg-black/30 text-white font-semibold text-xs rounded-xl transition-all duration-200 active:scale-95 print:hidden"
            >
              {t('reprocessButton')}
            </button>
          </div>
        </div>

        {/* Visual Diagnostic Panels Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
          
          {/* Diagnostic Imagery Columns */}
          <div className="lg:col-span-7 space-y-4">
            <GradCAMViewer
              originalImageB64={originalUrl || undefined}
              gradcamTier1B64={gradcamTier1Src}
              gradcamTier2B64={gradcamTier2Src}
            />
          </div>

          {/* Diagnostic Metrics, Conformal sets and Uncertainty Columns */}
          <div className="lg:col-span-5 space-y-6">
            <div>
              <h3 className="font-heading font-bold text-sm text-slate-800 dark:text-slate-200 mb-4 border-b border-slate-100 dark:border-slate-800 pb-2">
                {t('clinicalAnalysisParams')}
              </h3>

              {/* Grid Metrics */}
              <div className="space-y-4">
                
                {/* Confidence Level Badge */}
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/30 rounded-xl border border-slate-200/40 dark:border-slate-850">
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                    <Percent className="w-4 h-4 text-teal-500" />
                    {t('decisionConfidence')}
                  </span>
                  <span className={`text-sm font-extrabold font-heading ${
                    isPneumothorax ? 'text-red-500' : 'text-emerald-500'
                  }`}>
                    %{Math.round(prediction.confidence * 100)}
                  </span>
                </div>

                {/* Used Tier */}
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/30 rounded-xl border border-slate-200/40 dark:border-slate-850">
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                    <Shield className="w-4 h-4 text-teal-500" />
                    {t('labelTierUsed')}
                  </span>
                  <TierBadge tierUsed={prediction.tier_used} />
                </div>

                {/* Active Backbone Model */}
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/30 rounded-xl border border-slate-200/40 dark:border-slate-850">
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                    <HelpCircle className="w-4 h-4 text-teal-500" />
                    {t('labelModelName')}
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-600 dark:text-slate-400">
                    {prediction.model_version}
                  </span>
                </div>

                {/* Inference Duration */}
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/30 rounded-xl border border-slate-200/40 dark:border-slate-850">
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-teal-500" />
                    {t('labelInferenceTime')}
                  </span>
                  <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                    {prediction.inference_time_ms.toFixed(1)} ms
                  </span>
                </div>
              </div>
            </div>

            {/* Confidence Visualization Bar */}
            <ConfidenceBar value={prediction.confidence} label={t('decisionConfidence')} variant={isPneumothorax ? 'neutral' : 'positive'} />

            {/* Model Uncertainty Meter */}
            <UncertaintyBar mcVariance={prediction.mc_variance} mcPasses={prediction.mc_passes} />

            {/* Conformal Prediction Set Boundaries */}
            <ConformalSet members={prediction.conformal_set} coverage={prediction.conformal_coverage} />

          </div>
        </div>

        {/* Clinical Disclaimer Block */}
        <div className="bg-slate-50 dark:bg-slate-900/50 p-4 border-t border-slate-200/40 dark:border-slate-800/40 print:bg-white print:border-t-2 print:border-slate-400">
          <div className="flex items-start gap-2 text-amber-600 dark:text-amber-500 font-bold text-xs uppercase mb-1">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            {t('disclaimerTitle')}
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-relaxed text-justify">
            {t('disclaimerContent')}
          </p>
        </div>
      </div>
      
    </div>
  );
};
