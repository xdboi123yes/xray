/**
 * @file DicomUpload.tsx
 * @description Lightweight DICOM (.dcm) upload helper that POSTs to the backend
 * and surfaces the converted PNG preview returned by the API.
 */

import React, { useCallback, useState } from 'react';
import { FileScan, Loader2, AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface DicomUploadProps {
  endpoint?: string;
  onPredict?: (file: File) => void;
}

const MAX_BYTES = 15 * 1024 * 1024;

export const DicomUpload: React.FC<DicomUploadProps> = ({
  endpoint = '/api/v1/predict',
  onPredict,
}) => {
  const { t } = useTranslation();
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File | null) => {
      if (!file) return;
      setErrorMsg(null);
      if (file.size > MAX_BYTES) {
        setErrorMsg(t('uploadErrorSize', 'File size too large (max 15MB).'));
        return;
      }
      if (!file.name.toLowerCase().endsWith('.dcm')) {
        setErrorMsg(t('uploadDicomOnly', 'Please select a .dcm file.'));
        return;
      }
      try {
        setIsUploading(true);
        if (onPredict) {
          onPredict(file);
        } else {
          // Fire-and-forget upload for preview / probe purposes
          const fd = new FormData();
          fd.append('file', file);
          const res = await fetch(`${endpoint}?return_gradcam=false`, {
            method: 'POST',
            body: fd,
          });
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
        }
      } catch (err) {
        setErrorMsg(String(err));
      } finally {
        setIsUploading(false);
      }
    },
    [endpoint, onPredict, t],
  );

  return (
    <div className="border-2 border-dashed border-teal-500/30 rounded-2xl p-6 bg-teal-500/5 flex flex-col items-center gap-3 text-center">
      <FileScan className="w-8 h-8 text-teal-500" />
      <p className="text-sm font-bold text-slate-700 dark:text-slate-200">
        {t('dicomUploadTitle', 'DICOM (.dcm) upload')}
      </p>
      <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm">
        {t(
          'dicomUploadBody',
          'Drop a DICOM study or click below. Backend converts to PNG and runs the same tiered pipeline.',
        )}
      </p>

      <label className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-teal-500 hover:bg-teal-600 px-4 py-2 text-xs font-bold uppercase tracking-wider text-white transition-colors">
        {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileScan className="w-4 h-4" />}
        {isUploading ? t('uploading', 'Uploading...') : t('selectDicom', 'Select .dcm file')}
        <input
          type="file"
          accept=".dcm,application/dicom"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
      </label>

      {errorMsg && (
        <p className="inline-flex items-center gap-2 rounded-lg bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/30 px-3 py-1.5 text-[11px] font-semibold">
          <AlertCircle className="w-3.5 h-3.5" />
          {errorMsg}
        </p>
      )}
    </div>
  );
};
