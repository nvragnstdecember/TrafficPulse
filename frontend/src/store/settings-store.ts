import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Density = 'comfortable' | 'compact';

export interface SettingsState {
  /** Table/list row density. */
  density: Density;
  /** Default page size for paginated lists. */
  eventsPageSize: number;
  setDensity: (density: Density) => void;
  setEventsPageSize: (size: number) => void;
  reset: () => void;
}

const DEFAULTS = {
  density: 'comfortable' as Density,
  eventsPageSize: 25,
};

/** Persisted user preferences (H7B). */
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      setDensity: (density) => set({ density }),
      setEventsPageSize: (eventsPageSize) => set({ eventsPageSize }),
      reset: () => set({ ...DEFAULTS }),
    }),
    { name: 'trafficpulse-settings' },
  ),
);
