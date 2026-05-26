/**
 * @file UploadZone.tsx
 * @description Premium drag-and-drop radiograph selection zone featuring frosted glass,
 * clipboard paste listener support, dynamic file type validations, and HSL glow animations.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileImage, Clipboard, AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onFileSelected }) => {
  const { t } = useTranslation();
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [draggedFileName, setDraggedFileName] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Paste handler to process chest x-rays from clipboard directly
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.indexOf('image') !== -1) {
          const file = item.getAsFile();
          if (file) {
            validateAndSelectFile(file);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => {
      window.removeEventListener('paste', handlePaste);
    };
  }, []);

  const validateAndSelectFile = (file: File) => {
    setErrorMsg(null);
    setDraggedFileName(null);

    // Validate size (15MB max limit)
    const MAX_SIZE = 15 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      setErrorMsg(t('uploadErrorSize'));
      return;
    }

    // Accept standard images or simulate converting DICOM files
    const validExtensions = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom', 'image/webp'];
    const isDicom = file.name.toLowerCase().endsWith('.dcm') || file.type === 'application/dicom';
    const isValidType = validExtensions.includes(file.type) || isDicom;

    if (!isValidType) {
      setErrorMsg(t('uploadErrorType'));
      return;
    }

    // Trigger parent callback
    onFileSelected(file);
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setDraggedFileName(file.name);
      validateAndSelectFile(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
        className={`relative flex flex-col items-center justify-center min-h-[300px] p-8 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 glass-panel shadow-lg ${
          isDragActive
            ? 'border-teal-500 bg-teal-50/50 dark:bg-teal-950/20 scale-[1.01] shadow-teal-500/10'
            : 'border-slate-300 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-700 hover:bg-slate-100/50 dark:hover:bg-slate-900/10'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".png,.jpg,.jpeg,.dcm,image/png,image/jpeg,application/dicom"
          onChange={handleChange}
        />

        {/* Upload Visual Assets */}
        <div className={`p-4 rounded-full mb-4 transition-all duration-300 ${
          isDragActive 
            ? 'bg-teal-500 text-white pulse-ring-teal' 
            : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'
        }`}>
          {isDragActive ? (
            <Upload className="w-10 h-10 animate-bounce" />
          ) : (
            <FileImage className="w-10 h-10" />
          )}
        </div>

        {/* Dynamic Titles */}
        <h3 className="font-heading font-semibold text-lg text-slate-800 dark:text-slate-100 text-center mb-1">
          {draggedFileName ? draggedFileName : t('uploadTitle')}
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 text-center max-w-md mb-6 leading-relaxed">
          {t('uploadSubtitle')}
        </p>

        {/* Premium Shortcuts indicator */}
        <div className="flex items-center gap-6 text-xs text-slate-400 dark:text-slate-500 font-medium bg-slate-100/50 dark:bg-slate-900/50 py-2 px-4 rounded-full border border-slate-200/50 dark:border-slate-800/50">
          <span className="flex items-center gap-1.5">
            <Clipboard className="w-3.5 h-3.5" />
            {t('ctrlPaste')}
          </span>
          <span className="w-1 h-1 rounded-full bg-slate-300 dark:bg-slate-700"></span>
          <span>DICOM (.dcm) Uyumlu</span>
        </div>

        {/* Error Indicators */}
        {errorMsg && (
          <div className="absolute bottom-4 left-4 right-4 flex items-center gap-2 text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 rounded-xl p-3 text-xs animate-fade-in shadow-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="font-medium text-left leading-normal">{errorMsg}</span>
          </div>
        )}
      </div>
    </div>
  );
};
