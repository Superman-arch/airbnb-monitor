import { create } from 'zustand';

interface MonitoringState {
  isRecording: boolean;
  isPaused: boolean;
  motionDetectionEnabled: boolean;
  alertsEnabled: boolean;
  currentFPS: number;
  systemStatus: 'running' | 'stopped' | 'error';
  toggleRecording: () => void;
  togglePause: () => void;
  toggleMotionDetection: () => void;
  toggleAlerts: () => void;
  updateFPS: (fps: number) => void;
  setSystemStatus: (status: 'running' | 'stopped' | 'error') => void;
}

export const useMonitoringStore = create<MonitoringState>((set) => ({
  isRecording: false,
  isPaused: false,
  motionDetectionEnabled: true,
  alertsEnabled: true,
  currentFPS: 0,
  systemStatus: 'running',

  toggleRecording: () => set((state) => ({ 
    isRecording: !state.isRecording 
  })),

  togglePause: () => set((state) => ({ 
    isPaused: !state.isPaused 
  })),

  toggleMotionDetection: () => set((state) => ({ 
    motionDetectionEnabled: !state.motionDetectionEnabled 
  })),

  toggleAlerts: () => set((state) => ({ 
    alertsEnabled: !state.alertsEnabled 
  })),

  updateFPS: (fps: number) => set({ 
    currentFPS: fps 
  }),

  setSystemStatus: (status: 'running' | 'stopped' | 'error') => set({ 
    systemStatus: status 
  }),
}));