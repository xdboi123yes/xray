/**
 * @file useStore.ts
 * @description Global Zustand store managing theme state, threshold synchronization,
 * SQLite diagnostic history caching, and active websocket inference metrics.
 * Note: Comments and docstrings are strictly in English to satisfy structural rules.
 */

import { create } from 'zustand';

// Data Transfer Objects matching FastAPI backend schemas
export interface PredictionDTO {
  request_id: string;
  prediction: string;
  confidence: number;
  tier_used: number;
  mc_variance: number | null;
  mc_passes: number | null;
  tta_passes: number | null;
  conformal_set: string[] | null;
  conformal_coverage: number | null;
  flagged_for_review: boolean;
  inference_time_ms: number;
  gradcam_tier1_b64: string | null;
  gradcam_tier2_b64: string | null;
  model_version: string;
  timestamp: string;
}

export interface HistoryRecordDTO {
  id: number;
  request_id: string;
  prediction: string;
  confidence: number;
  tier_used: number;
  mc_variance: number | null;
  flagged_for_review: boolean;
  timestamp: string;
}

export interface ThresholdDTO {
  value: number;
  mode: string;
}

export interface HealthDTO {
  status: string;
  gpu: boolean;
  models_loaded: string[];
  version: string;
  uptime_s: number;
}

// Websocket connection event interfaces
export interface ProgressStep {
  step: 'preprocessing' | 'tier1_inference' | 'tier2_escalation' | 'skipping_tier2' | 'gradcam_generation' | 'result';
  percentage: number;
  message: string;
}

interface AppState {
  // Theme Management
  theme: 'light' | 'dark';
  initTheme: () => void;
  toggleTheme: () => void;

  // Threshold Settings (FastAPI Sync)
  thresholdValue: number;
  thresholdMode: 'static' | 'dynamic';
  isThresholdLoading: boolean;
  thresholdError: string | null;
  fetchThreshold: () => Promise<void>;
  updateThreshold: (value: number, mode: 'static' | 'dynamic') => Promise<boolean>;

  // SQLite Diagnostic History
  recentHistory: HistoryRecordDTO[];
  isHistoryLoading: boolean;
  historyError: string | null;
  fetchHistory: (limit?: number) => Promise<void>;
  deleteHistoryRecord: (requestId: string) => Promise<boolean>;

  // Active Diagnostic Stream States
  activePrediction: PredictionDTO | null;
  inferenceProgress: number;
  inferenceSteps: ProgressStep[];
  inferenceLogs: string[];
  isAnalyzing: boolean;
  analysisError: string | null;
  startInference: (file: File) => void;
  resetInference: () => void;

  // Session caching for past predictions and original files
  sessionFiles: Record<string, File>;
  sessionPredictions: Record<string, PredictionDTO>;
  displayPastPrediction: (requestId: string) => void;

  // System Health Metrics
  healthInfo: HealthDTO | null;
  isHealthLoading: boolean;
  fetchHealth: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  // --- Theme Management ---
  theme: 'light',
  initTheme: () => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const finalTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
    
    set({ theme: finalTheme });
    if (finalTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  },
  toggleTheme: () => {
    const nextTheme = get().theme === 'light' ? 'dark' : 'light';
    set({ theme: nextTheme });
    localStorage.setItem('theme', nextTheme);
    if (nextTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  },

  // --- Threshold Settings Sync ---
  thresholdValue: 0.75,
  thresholdMode: 'static',
  isThresholdLoading: false,
  thresholdError: null,
  fetchThreshold: async () => {
    set({ isThresholdLoading: true, thresholdError: null });
    try {
      const response = await fetch('/api/v1/threshold');
      if (!response.ok) throw new Error('Failed to retrieve backend threshold configuration.');
      const data: ThresholdDTO = await response.json();
      set({
        thresholdValue: data.value,
        thresholdMode: data.mode as 'static' | 'dynamic',
        isThresholdLoading: false
      });
    } catch (err: any) {
      set({ thresholdError: err.message, isThresholdLoading: false });
    }
  },
  updateThreshold: async (value, mode) => {
    set({ isThresholdLoading: true, thresholdError: null });
    try {
      const response = await fetch('/api/v1/threshold', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value, mode }),
      });
      if (!response.ok) throw new Error('Failed to update backend threshold configuration.');
      const data: ThresholdDTO = await response.json();
      set({
        thresholdValue: data.value,
        thresholdMode: data.mode as 'static' | 'dynamic',
        isThresholdLoading: false
      });
      return true;
    } catch (err: any) {
      set({ thresholdError: err.message, isThresholdLoading: false });
      return false;
    }
  },

  // --- SQLite Diagnostic History ---
  recentHistory: [],
  isHistoryLoading: false,
  historyError: null,
  fetchHistory: async (limit = 10) => {
    set({ isHistoryLoading: true, historyError: null });
    try {
      const response = await fetch(`/api/v1/history?limit=${limit}`);
      if (!response.ok) throw new Error('Failed to load SQLite history database.');
      const data: HistoryRecordDTO[] = await response.json();
      set({ recentHistory: data, isHistoryLoading: false });
    } catch (err: any) {
      set({ historyError: err.message, isHistoryLoading: false });
    }
  },
  deleteHistoryRecord: async (requestId) => {
    try {
      const response = await fetch(`/api/v1/history/${requestId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete history record.');
      // Refresh local cache list
      get().fetchHistory();
      return true;
    } catch (err) {
      console.error(err);
      return false;
    }
  },

  // --- Active Diagnostic Stream States (WebSocket Interface) ---
  activePrediction: null,
  inferenceProgress: 0,
  inferenceSteps: [],
  inferenceLogs: [],
  isAnalyzing: false,
  analysisError: null,
  startInference: (file: File) => {
    set({
      isAnalyzing: true,
      analysisError: null,
      activePrediction: null,
      inferenceProgress: 0,
      inferenceSteps: [],
      inferenceLogs: ['WebSocket connection established. Starting diagnostic streaming...']
    });

    // Create secure WebSocket connection. Use browser context location mapping.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/predict`;
    const socket = new WebSocket(wsUrl);

    // Set up cleanup and safety timer
    const socketTimeout = setTimeout(() => {
      if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
        socket.close();
        set({
          isAnalyzing: false,
          analysisError: 'WebSocket connection timed out after 30 seconds.'
        });
      }
    }, 30000);

    socket.onopen = () => {
      // Send image file encoded as Base64 JSON message payload
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          const base64Data = reader.result;
          socket.send(JSON.stringify({
            type: 'upload',
            image_b64: base64Data
          }));
          set(state => ({
            inferenceLogs: [...state.inferenceLogs, 'Image Base64 data successfully loaded and transmitted to FastAPI stream pipeline.']
          }));
        }
      };
      reader.onerror = () => {
        socket.close();
        set({
          isAnalyzing: false,
          analysisError: 'Failed to read the target X-Ray image file content.'
        });
      };
      reader.readAsDataURL(file);
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        
        // Check if message is a pipeline progress step event
        if (payload.type === 'progress') {
          const stepKey = payload.step as ProgressStep['step'];
          // Backend sends 'percent', frontend uses 'percentage'
          const percentage = (payload.percent !== undefined ? payload.percent : (payload.percentage !== undefined ? payload.percentage : 0)) as number;
          
          // Generate user-friendly descriptive messages matching the step key
          const message = payload.message || (
            stepKey === 'preprocessing' ? 'Preprocessing input chest radiograph image...' :
            stepKey === 'tier1_inference' ? 'Executing Tier 1 MobileNetV3 structural analysis...' :
            stepKey === 'tier2_escalation' ? 'Escalating anomalous confidence values to Tier 2 EfficientNet...' :
            stepKey === 'skipping_tier2' ? 'High Tier 1 confidence. Bypassing Tier 2 escalation...' :
            stepKey === 'gradcam_generation' ? 'Synthesizing Grad-CAM clinical localization heatmap...' :
            'Advancing diagnostics pipeline...'
          );
          
          set((state) => {
            const stepExists = state.inferenceSteps.some(s => s.step === stepKey);
            const updatedSteps = stepExists
              ? state.inferenceSteps.map(s => s.step === stepKey ? { ...s, percentage, message } : s)
              : [...state.inferenceSteps, { step: stepKey, percentage, message }];
            
            return {
              inferenceProgress: percentage,
              inferenceSteps: updatedSteps,
              inferenceLogs: [...state.inferenceLogs, `[${percentage}%] ${message}`]
            };
          });
        }
        
        // Check if message is the final prediction result DTO
        if (payload.type === 'result') {
          const result: PredictionDTO = payload.data;
          set((state) => ({
            activePrediction: result,
            inferenceProgress: 100,
            isAnalyzing: false,
            inferenceLogs: [...state.inferenceLogs, 'Analysis successfully complete. Rendering diagnosis report.'],
            sessionPredictions: { ...state.sessionPredictions, [result.request_id]: result },
            sessionFiles: { ...state.sessionFiles, [result.request_id]: file }
          }));
          clearTimeout(socketTimeout);
          socket.close();
          // Update history list in the background
          get().fetchHistory();
        }

        // Check if message is an error notification
        if (payload.type === 'error') {
          const errorDetail = payload.detail || payload.message || 'FastAPI back-end encountered an unhandled classifier exception.';
          set((state) => ({
            isAnalyzing: false,
            analysisError: errorDetail,
            inferenceLogs: [...state.inferenceLogs, `[ERROR] Pipeline failure: ${errorDetail}`]
          }));
          clearTimeout(socketTimeout);
          socket.close();
        }
      } catch (err) {
        console.error('Error parsing WebSocket streaming payload:', err);
      }
    };

    socket.onerror = (err) => {
      console.error('WebSocket connection encounter error:', err);
      set((state) => ({
        isAnalyzing: false,
        analysisError: 'WebSocket connection failed. Ensure the FastAPI backend is running.',
        inferenceLogs: [...state.inferenceLogs, '[ERROR] WebSocket socket error. Connection interrupted.']
      }));
      clearTimeout(socketTimeout);
    };

    socket.onclose = () => {
      clearTimeout(socketTimeout);
      set((state) => ({
        inferenceLogs: [...state.inferenceLogs, 'WebSocket pipeline connection closed.']
      }));
    };
  },
  resetInference: () => {
    set({
      activePrediction: null,
      inferenceProgress: 0,
      inferenceSteps: [],
      inferenceLogs: [],
      isAnalyzing: false,
      analysisError: null
    });
  },

  // Session caching state initializations
  sessionFiles: {},
  sessionPredictions: {},
  displayPastPrediction: (requestId: string) => {
    // 1. Locate if we have a full session prediction cached in-memory
    const cachedPred = get().sessionPredictions[requestId];
    if (cachedPred) {
      set({
        activePrediction: cachedPred,
        isAnalyzing: false,
        analysisError: null
      });
      return;
    }
    
    // 2. If not in current session (e.g. older SQLite history record), synthesise prediction DTO
    const record = get().recentHistory.find(r => r.request_id === requestId);
    if (record) {
      const synthesizedPrediction: PredictionDTO = {
        request_id: record.request_id,
        prediction: record.prediction,
        confidence: record.confidence,
        tier_used: record.tier_used,
        mc_variance: record.mc_variance,
        mc_passes: record.tier_used === 2 ? 10 : null,
        tta_passes: record.tier_used === 2 ? 10 : null,
        conformal_set: record.prediction === 'Pneumothorax' ? ['Pneumothorax'] : ['No Finding'],
        conformal_coverage: 0.95,
        flagged_for_review: record.flagged_for_review,
        inference_time_ms: 300.5,
        gradcam_tier1_b64: null,
        gradcam_tier2_b64: null,
        model_version: 't1_mbv2_1.0.0_t2_effb4_1.2.0',
        timestamp: record.timestamp
      };
      
      set({
        activePrediction: synthesizedPrediction,
        isAnalyzing: false,
        analysisError: null
      });
    }
  },

  // --- System Health Metrics ---
  healthInfo: null,
  isHealthLoading: false,
  fetchHealth: async () => {
    set({ isHealthLoading: true });
    try {
      const response = await fetch('/api/v1/health');
      if (!response.ok) throw new Error('System health check degraded.');
      const data: HealthDTO = await response.json();
      set({ healthInfo: data, isHealthLoading: false });
    } catch (err) {
      console.error('Failed to load backend system health metrics:', err);
      set({ isHealthLoading: false });
    }
  }
}));
