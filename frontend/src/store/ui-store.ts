import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UiState {
  /** Desktop rail collapsed to icons-only. Persisted. */
  sidebarCollapsed: boolean;
  /** Mobile drawer open. Transient (never persisted). */
  mobileSidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setMobileSidebarOpen: (open: boolean) => void;
}

/**
 * UI shell state (H7B): sidebar collapse + mobile drawer. Only the desktop
 * collapse preference is persisted; the mobile drawer always starts closed.
 */
export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      mobileSidebarOpen: false,
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setMobileSidebarOpen: (open) => set({ mobileSidebarOpen: open }),
    }),
    {
      name: 'trafficpulse-ui',
      partialize: (state) => ({ sidebarCollapsed: state.sidebarCollapsed }),
    },
  ),
);
