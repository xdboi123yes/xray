/**
 * @file BatchUploader.tsx
 * @description Multi-file uploader that calls the `/predict/batch` endpoint and
 * surfaces per-file progress and a download-as-CSV summary.
 */

import React, { useCallback, useState } from 'react';
import { Files, Loader2, Download } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface BatchUploaderProps {
  endpoint?: string;
  maxFiles?: number;
}

interface BatchResult {
  request_id: string;
  prediction: string;
  confidence: number;
  tier_used: number;
}

export const BatchUploader: React.FC<BatchUploaderProps> = ({
  endpoint = '/api/v1/predict/batch',
  maxFiles = 50,
}) => {
  const { t } = useTranslation();
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list) return;
      const arr = Array.from(list).slice(0, maxFiles);
      setFiles(arr);
      setResults([]);
      setErrorMsg(null);
    },
    [maxFiles],
  );

  const runBatch = useCallback(async () => {
    if (files.length === 0) return;
    setIsProcessing(true);
    setErrorMsg(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append('files', f));
      const res = await fetch(endpoint, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BatchResult[] = await res.json();
      setResults(data);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setIsProcessing(false);
    }
  }, [endpoint, files]);

  const downloadCsv = useCallback(() => {
    const header = 'request_id,prediction,confidence,tier_used\n';
    const body = results
      .map((r) => `${r.request_id},${r.prediction},${r.confidence.toFixed(4)},${r.tier_used}`)
      .join('\n');
    const blob = new Blob([header + body], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [results]);

  return (
    <div className="rounded-2xl border border-slate-200/60 dark:border-slate-800/60 bg-white/50 dark:bg-slate-900/30 p-5 space-y-4">
      <div className="flex items-center gap-2 text-sm font-bold text-slate-700 dark:text-slate-200">
        <Files className="w-4 h-4 text-teal-500" />
        {t('batchUploadTitle', 'Batch processing')}
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest ml-auto">
          {t('batchMaxFiles', 'Max {{n}} files', { n: maxFiles })}
        </span>
      </div>

      <label className="cursor-pointer flex items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-900/30 px-4 py-6 text-xs font-semibold text-slate-500 dark:text-slate-400 hover:border-teal-500/50 transition-colors">
        <Files className="w-4 h-4" />
        {t('batchPickFiles', 'Click to choose PNG/JPEG/DICOM files')}
        <input
          type="file"
          multiple
          accept="image/*,.dcm,application/dicom"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </label>

      {files.length > 0 && (
        <div className="text-xs text-slate-500 dark:text-slate-400 font-medium">
          {t('batchSelectedCount', '{{n}} files selected', { n: files.length })}
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={files.length === 0 || isProcessing}
          onClick={runBatch}
          className="inline-flex items-center gap-2 rounded-xl bg-teal-500 hover:bg-teal-600 disabled:bg-slate-300 dark:disabled:bg-slate-800 disabled:cursor-not-allowed px-4 py-2 text-xs font-bold uppercase tracking-wider text-white transition-colors"
        >
          {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Files className="w-4 h-4" />}
          {isProcessing ? t('batchRunning', 'Running...') : t('batchRun', 'Run batch')}
        </button>
        {results.length > 0 && (
          <button
            type="button"
            onClick={downloadCsv}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-slate-700 px-3 py-2 text-[11px] font-bold uppercase tracking-wider text-slate-700 dark:text-slate-200 hover:bg-slate-100/60 dark:hover:bg-slate-800/40 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            {t('batchDownloadCsv', 'Download CSV')}
          </button>
        )}
      </div>

      {errorMsg && (
        <div className="rounded-lg bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/30 px-3 py-2 text-xs">
          {errorMsg}
        </div>
      )}

      {results.length > 0 && (
        <div className="max-h-72 overflow-y-auto rounded-xl border border-slate-200/60 dark:border-slate-800/60 divide-y divide-slate-100 dark:divide-slate-800/40">
          {results.map((r) => (
            <div
              key={r.request_id}
              className="flex items-center justify-between px-4 py-2 text-xs font-medium"
            >
              <span className="font-mono text-slate-400 truncate w-32">{r.request_id.slice(0, 8)}</span>
              <span className="font-bold text-slate-700 dark:text-slate-200">{r.prediction}</span>
              <span className="text-slate-500">{(r.confidence * 100).toFixed(1)}%</span>
              <span className="text-[10px] uppercase tracking-wider text-teal-500">T{r.tier_used}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
