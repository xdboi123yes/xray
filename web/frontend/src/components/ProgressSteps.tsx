/**
 * @file ProgressSteps.tsx
 * @description Real-time diagnostic pipeline progression indicator.
 * Displays interactive steps synced with WebSocket states and an auto-scrolling CLI log console.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useEffect, useRef } from 'react';
import { Terminal, CheckCircle2, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { ProgressStep } from '../store/useStore';

interface ProgressStepsProps {
  progress: number;
  steps: ProgressStep[];
  logs: string[];
  currentStepKey?: 'preprocessing' | 'tier1_inference' | 'tier2_escalation' | 'skipping_tier2' | 'gradcam_generation' | 'result';
}

export const ProgressSteps: React.FC<ProgressStepsProps> = ({ progress, steps, logs }) => {
  const { t } = useTranslation();
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll logs console to bottom on new updates
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Define static diagnostic steps mapping for standard visualization
  const standardSteps = [
    { key: 'preprocessing', label: t('stepPrep'), desc: t('stepPrepDesc'), activeVal: 20 },
    { key: 'tier1_inference', label: t('stepTier1'), desc: t('stepTier1Desc'), activeVal: 40 },
    { key: 'tier2_escalation', label: t('stepTier2'), desc: t('stepTier2Desc'), activeVal: 70 },
    { key: 'gradcam_generation', label: t('stepGradcam'), desc: t('stepGradcamDesc'), activeVal: 90 },
    { key: 'result', label: t('stepResult'), desc: t('stepResultDesc'), activeVal: 100 }
  ];

  // Helper to determine active step state
  const getStepState = (stepKey: string) => {
    // Locate if this step has been reported from the WS backend stream
    const backendReportedStep = steps.find(s => s.step === stepKey);
    const isTier2Skipped = steps.some(s => s.step === 'skipping_tier2');

    if (stepKey === 'tier2_escalation' && isTier2Skipped) {
      return {
        isCompleted: true,
        isActive: false,
        label: t('stepTier2Skipped'),
        desc: t('stepTier2SkippedDesc')
      };
    }

    if (backendReportedStep) {
      const isCompleted = progress > backendReportedStep.percentage || progress === 100;
      const isActive = progress === backendReportedStep.percentage && progress < 100;
      return { isCompleted, isActive, label: undefined, desc: undefined };
    }

    // Fallback based on relative percentage levels
    let stepVal = 20;
    if (stepKey === 'preprocessing') stepVal = 20;
    else if (stepKey === 'tier1_inference') stepVal = 40;
    else if (stepKey === 'tier2_escalation') stepVal = 70;
    else if (stepKey === 'gradcam_generation') stepVal = 90;
    else if (stepKey === 'result') stepVal = 100;

    const isCompleted = progress > stepVal || progress === 100;
    const isActive = progress >= stepVal - 15 && progress <= stepVal && progress < 100;

    return { isCompleted, isActive, label: undefined, desc: undefined };
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-8 animate-fade-in">
      
      {/* Real-time Percentage Indicator */}
      <div className="flex flex-col items-center justify-center text-center">
        <div className="relative flex items-center justify-center w-36 h-36 rounded-full border-4 border-slate-200/40 dark:border-slate-800/40 bg-white/5 dark:bg-slate-900/5 shadow-inner">
          <div className="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin" style={{ animationDuration: '3s' }}></div>
          <div className="flex flex-col items-center justify-center">
            <span className="font-heading font-extrabold text-4xl text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-emerald-500 select-none filter drop-shadow-[0_2px_10px_rgba(20,184,166,0.2)]">
              %{progress}
            </span>
            <span className="text-[10px] tracking-widest font-semibold text-slate-400 dark:text-slate-500 uppercase mt-1">
              {t('activeInference')}
            </span>
          </div>
        </div>
      </div>

      {/* Modern Pipeline Flow Line */}
      <div className="relative glass-panel rounded-2xl p-6 shadow-md border border-slate-200/50 dark:border-slate-800/50">
        <div className="absolute top-[38px] left-[44px] right-[44px] h-[3px] bg-slate-200 dark:bg-slate-800 -z-10 rounded-full hidden md:block">
          <div 
            className="h-full bg-gradient-to-r from-teal-500 to-emerald-500 rounded-full transition-all duration-500 animate-flow"
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          {standardSteps.map((step) => {
            const { isCompleted, isActive, label, desc } = getStepState(step.key);
            const displayLabel = label || step.label;
            const displayDesc = desc || step.desc;

            return (
              <div key={step.key} className="flex flex-row md:flex-col items-start md:items-center text-left md:text-center group">
                {/* Stepper Node Bubble */}
                <div className="mr-4 md:mr-0 md:mb-3 shrink-0">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                    isCompleted
                      ? 'bg-emerald-500 border-emerald-500 text-white shadow-lg shadow-emerald-500/20'
                      : isActive
                      ? 'bg-teal-500 border-teal-500 text-white pulse-ring-teal'
                      : 'bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-800 text-slate-400 dark:text-slate-600'
                  }`}>
                    {isCompleted ? (
                      <CheckCircle2 className="w-5 h-5" />
                    ) : isActive ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <span className="text-xs font-bold font-heading">{step.activeVal}%</span>
                    )}
                  </div>
                </div>

                {/* Text Content */}
                <div>
                  <h4 className={`font-heading font-semibold text-xs transition-colors duration-200 ${
                    isActive ? 'text-teal-600 dark:text-teal-400 font-bold' : isCompleted ? 'text-slate-800 dark:text-slate-200' : 'text-slate-400 dark:text-slate-600'
                  }`}>
                    {displayLabel}
                  </h4>
                  <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5 leading-relaxed hidden md:block max-w-[120px] mx-auto">
                    {displayDesc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Streaming Console Log Monitor */}
      <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 text-slate-300 shadow-xl">
        {/* Terminal Header */}
        <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800 text-xs font-mono font-medium text-slate-400">
          <span className="flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-teal-500" />
            {t('fastApiConsole')}
          </span>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-green-500/80"></span>
          </div>
        </div>

        {/* Console logs box */}
        <div className="p-4 h-48 overflow-y-auto font-mono text-[11px] leading-relaxed text-left space-y-1.5 scrollbar-thin scrollbar-thumb-slate-800">
          {logs.map((log, index) => {
            const isError = log.includes('[ERROR]');
            const isSuccess = log.includes('successfully') || log.includes('Complete') || log.includes('100%');
            return (
              <div 
                key={index} 
                className={`transition-all duration-150 ${
                  isError ? 'text-red-400' : isSuccess ? 'text-teal-400' : 'text-slate-300'
                }`}
              >
                <span className="text-slate-600 select-none mr-2">xray-sys:~$</span>
                {log}
              </div>
            );
          })}
          <div ref={terminalEndRef} />
        </div>
      </div>
      
    </div>
  );
};
