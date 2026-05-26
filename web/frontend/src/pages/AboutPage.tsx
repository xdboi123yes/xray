/**
 * @file AboutPage.tsx
 * @description Renders te thesis academic context, cascading decision tree, and conformal bounds specifications.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import { BookOpen, ShieldCheck, Cpu, Award, Zap, HelpCircle } from 'lucide-react';

export const AboutPage: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="space-y-8 animate-fade-in text-slate-700 dark:text-slate-300">
      
      {/* Page Header */}
      <div>
        <h2 className="font-heading font-extrabold text-3xl text-slate-900 dark:text-white tracking-tight flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-teal-500" />
          {t('aboutPageTitle', 'Clinical Mechanics & Thesis Context')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Learn about the multi-stage cascading pipeline, conformal prediction guarantees, and Bachelor Thesis architecture.
        </p>
      </div>

      {/* Grid: Academic Info & System Specifications */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Thesis Details Card */}
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-6 rounded-3xl backdrop-blur-xl space-y-6">
          <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-teal-500" />
            Bachelor Thesis Details
          </h3>
          <div className="space-y-4 text-xs font-semibold">
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Institution</span>
              <span className="text-slate-900 dark:text-white">"1 Decembrie 1918" University of Alba Iulia</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Defense Date</span>
              <span className="text-slate-900 dark:text-white">Q3 2026</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Subject</span>
              <span className="text-slate-900 dark:text-white">Tiered Clinical Decision Support for Pneumothorax</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-400 uppercase tracking-wider">Methodology</span>
              <span className="text-teal-600 dark:text-teal-400">Cascading Screening + Conformal Inference</span>
            </div>
          </div>
        </div>

        {/* Core Specs Card */}
        <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-6 rounded-3xl backdrop-blur-xl space-y-6">
          <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-teal-500" />
            Cascading Model Parameters
          </h3>
          <div className="space-y-4 text-xs font-semibold">
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Tier 1 Backbone</span>
              <span className="text-slate-900 dark:text-white">MobileNetV2 (Parameters: ~2.2M)</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Tier 2 Backbone</span>
              <span className="text-slate-900 dark:text-white">Ark+ / EfficientNetB4 (Parameters: ~19.3M)</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-200/30 dark:border-slate-800/30">
              <span className="text-slate-400 uppercase tracking-wider">Nominal Conformal Level</span>
              <span className="text-slate-900 dark:text-white">1 - alpha = 0.95 (95% Coverage)</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-400 uppercase tracking-wider">Uncertainty Estimation</span>
              <span className="text-slate-900 dark:text-white">Monte Carlo Dropout (passes = 20)</span>
            </div>
          </div>
        </div>

      </div>

      {/* Decision cascade details */}
      <div className="bg-white/50 dark:bg-slate-900/30 border border-slate-200/50 dark:border-slate-800/50 p-8 rounded-3xl backdrop-blur-xl space-y-6">
        <h3 className="font-heading font-extrabold text-xl text-slate-900 dark:text-white flex items-center gap-2">
          <ShieldCheck className="w-6 h-6 text-teal-500" />
          The Tiered Clinical Cascade Mechanism
        </h3>
        
        <div className="space-y-4 leading-relaxed font-semibold text-xs text-slate-500 dark:text-slate-400">
          <p>
            In clinical emergency settings (like critical Pneumothorax detection), latency and accuracy represent life-or-death tradeoffs. The proposed architecture cascadingly delegates decisions between a highly efficient screening layer (Tier 1 MobileNetV2) and a deep specialist layer (Tier 2 Ark+) based on predictive uncertainty boundaries:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 text-slate-700 dark:text-slate-300">
            <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl space-y-2">
              <div className="flex items-center gap-2 text-teal-600 dark:text-teal-400 font-extrabold">
                <Zap className="w-4 h-4" />
                1. Tier 1 Fast Scan
              </div>
              <p className="text-[11px] font-normal leading-relaxed text-slate-500 dark:text-slate-400">
                MobileNetV2 processes the input radiograph. If the predictive probability falls in high-confidence zones (close to 0.0 or 1.0), the decision is finalized instantly, achieving Sub-10ms diagnostic loops.
              </p>
            </div>

            <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl space-y-2">
              <div className="flex items-center gap-2 text-amber-500 font-extrabold">
                <HelpCircle className="w-4 h-4" />
                2. Routing Gate
              </div>
              <p className="text-[11px] font-normal leading-relaxed text-slate-500 dark:text-slate-400">
                If Tier 1 output lands in the ambiguous decision region (e.g. probability near threshold \theta), the sample is escalated to the Specialist Tier 2 or flagged as OOD (Out-of-Distribution).
              </p>
            </div>

            <div className="p-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-250/20 rounded-2xl space-y-2">
              <div className="flex items-center gap-2 text-purple-500 font-extrabold">
                <ShieldCheck className="w-4 h-4" />
                3. Tier 2 Conformal Set
              </div>
              <p className="text-[11px] font-normal leading-relaxed text-slate-500 dark:text-slate-400">
                Tier 2 specialists process the X-ray with MC Dropout/TTA. Conformal sets guarantee that the true clinical condition is represented in the prediction output set at the nominal 95% confidence bounds.
              </p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};
