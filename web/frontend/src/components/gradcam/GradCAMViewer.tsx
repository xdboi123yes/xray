/**
 * @file GradCAMViewer.tsx
 * @description Interactive visualizer for Tier 1 and Tier 2 Grad-CAM diagnostic heatmaps.
 * Implements absolute overlay positioning with opacity scaling and side-by-side reviews.
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sliders, Eye, EyeOff, LayoutGrid } from 'lucide-react';

interface GradCAMViewerProps {
  originalImageB64?: string; // Original base64 or URL
  gradcamTier1B64?: string;  // Tier 1 Base64 heatmap
  gradcamTier2B64?: string;  // Tier 2 Base64 heatmap
}

export const GradCAMViewer: React.FC<GradCAMViewerProps> = ({
  originalImageB64,
  gradcamTier1B64,
  gradcamTier2B64,
}) => {
  const { t } = useTranslation();
  const [opacity, setOpacity] = useState<number>(0.6);
  const [activeTier, setActiveTier] = useState<1 | 2>(gradcamTier2B64 ? 2 : 1);
  const [sideBySide, setSideBySide] = useState<boolean>(false);
  const [hideHeatmap, setHideHeatmap] = useState<boolean>(false);

  const activeHeatmap = activeTier === 2 ? gradcamTier2B64 : gradcamTier1B64;
  const hasBoth = !!(gradcamTier1B64 && gradcamTier2B64);

  // If no original image, render a beautiful fallback placeholder
  if (!originalImageB64) {
    return (
      <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-200/50 dark:border-slate-800/50 rounded-3xl bg-slate-50/50 dark:bg-slate-950/20">
        <EyeOff className="w-8 h-8 text-slate-400 mb-2" />
        <span className="text-xs font-bold text-slate-450 uppercase tracking-wider">{t('heatmapUnavailable')}</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      
      {/* Interactive Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-slate-50/50 dark:bg-slate-950/20 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl">
        
        {/* Tier Toggler (if both are present) */}
        <div className="flex gap-2">
          {hasBoth && (
            <>
              <button
                type="button"
                onClick={() => setActiveTier(1)}
                className={`py-1.5 px-3 rounded-xl font-bold text-[10px] uppercase tracking-wider border transition-all ${
                  activeTier === 1
                    ? 'bg-teal-500 text-white border-teal-500'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-500 hover:bg-slate-50'
                }`}
              >
                Tier 1 Map
              </button>
              <button
                type="button"
                onClick={() => setActiveTier(2)}
                className={`py-1.5 px-3 rounded-xl font-bold text-[10px] uppercase tracking-wider border transition-all ${
                  activeTier === 2
                    ? 'bg-teal-500 text-white border-teal-500'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-500 hover:bg-slate-50'
                }`}
              >
                Tier 2 Map
              </button>
            </>
          )}
        </div>

        {/* View Mode Selectors */}
        <div className="flex items-center gap-3">
          
          {/* Hide Heatmap Toggle */}
          <button
            type="button"
            onClick={() => setHideHeatmap(!hideHeatmap)}
            className="flex items-center gap-1.5 py-1.5 px-3 border border-slate-200 dark:border-slate-800 rounded-xl text-[10px] font-bold uppercase tracking-wider text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors"
          >
            {hideHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            {hideHeatmap ? 'Show Heatmap' : 'Hide Heatmap'}
          </button>

          {/* Side-by-Side Review */}
          {activeHeatmap && (
            <button
              type="button"
              onClick={() => setSideBySide(!sideBySide)}
              className={`flex items-center gap-1.5 py-1.5 px-3 border rounded-xl text-[10px] font-bold uppercase tracking-wider transition-all ${
                sideBySide
                  ? 'bg-teal-500 text-white border-teal-500'
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-500 hover:bg-slate-50'
              }`}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              {t('sideBySideReview')}
            </button>
          )}
        </div>
      </div>

      {/* Opacity Overlay Slider */}
      {!hideHeatmap && !sideBySide && activeHeatmap && (
        <div className="flex items-center gap-4 p-4 bg-slate-50/50 dark:bg-slate-950/20 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl">
          <Sliders className="w-4 h-4 text-teal-500 shrink-0" />
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest whitespace-nowrap">
            {t('opacitySlider')}: {(opacity * 100).toFixed(0)}%
          </span>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-500"
          />
        </div>
      )}

      {/* Main Visual Display */}
      {sideBySide && activeHeatmap ? (
        
        /* Side-by-Side Mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="relative border border-slate-200/50 dark:border-slate-800/50 rounded-3xl overflow-hidden bg-black flex items-center justify-center aspect-square shadow-lg">
            <img src={originalImageB64} alt="Original Chest X-Ray" className="w-full h-full object-cover" />
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-xl text-[9px] font-bold text-white uppercase tracking-wider">
              {t('tabOriginal')}
            </div>
          </div>

          <div className="relative border border-slate-200/50 dark:border-slate-800/50 rounded-3xl overflow-hidden bg-black flex items-center justify-center aspect-square shadow-lg">
            <img src={originalImageB64} alt="Original X-Ray" className="w-full h-full object-cover" />
            <img
              src={activeHeatmap}
              alt="Grad-CAM Saliency Overlay"
              className="absolute inset-0 w-full h-full object-cover mix-blend-jetpack" // absolute cover overlay
              style={{ opacity: 0.8 }}
            />
            <div className="absolute top-4 left-4 bg-teal-500/80 backdrop-blur-md px-2.5 py-1 rounded-xl text-[9px] font-bold text-white uppercase tracking-wider">
              Grad-CAM (Tier {activeTier})
            </div>
          </div>
        </div>
      ) : (
        
        /* Overlay / Single Mode */
        <div className="relative border border-slate-200/50 dark:border-slate-800/50 rounded-3xl overflow-hidden bg-black flex items-center justify-center aspect-square max-w-lg mx-auto shadow-2xl">
          <img
            src={originalImageB64}
            alt="Original Chest X-Ray"
            className="w-full h-full object-cover"
          />
          {!hideHeatmap && activeHeatmap && (
            <img
              src={activeHeatmap}
              alt="Grad-CAM Heatmap"
              className="absolute inset-0 w-full h-full object-cover transition-opacity duration-150"
              style={{ opacity: opacity }}
            />
          )}
          <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-xl text-[9px] font-bold text-white uppercase tracking-wider">
            {!hideHeatmap && activeHeatmap ? `Grad-CAM Attributions (Tier ${activeTier})` : t('tabOriginal')}
          </div>
        </div>
      )}

      {/* Saliency Legend */}
      {!hideHeatmap && activeHeatmap && (
        <div className="p-4 bg-slate-50/50 dark:bg-slate-950/20 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl space-y-2 max-w-lg mx-auto">
          <div className="flex items-center justify-between text-[9px] font-bold text-slate-400 uppercase tracking-widest">
            <span>Low Suspicion</span>
            <span>Clinical Saliency Spectrum</span>
            <span>High Risk (Pneumothorax)</span>
          </div>
          <div className="h-2.5 w-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500 rounded-full border border-slate-200/20"></div>
        </div>
      )}

    </div>
  );
};
