import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { type EventFilters, type WorkspaceSort, DEFAULT_EVENT_FILTERS } from '@/lib/workspace';

interface WorkspacePrefsState {
  /** The event-list filters (the single source of truth; persisted). */
  filters: EventFilters;
  /** The event-list sort order (persisted). */
  sort: WorkspaceSort;
  /** Whether the multi-select (bulk actions) mode is on. */
  selectionMode: boolean;

  setFilters: (filters: EventFilters) => void;
  setSort: (sort: WorkspaceSort) => void;
  setSelectionMode: (on: boolean) => void;
  resetFilters: () => void;
}

/**
 * Persisted workspace preferences (H7E).
 *
 * The event-list filters and sort live here (not as component state) so they are
 * a single source of truth, survive a refresh, and are shared without prop
 * drilling — the "remember filters / remember workspace preferences" goal. The
 * list components stay presentational and receive these via the workspace.
 */
export const useWorkspacePrefsStore = create<WorkspacePrefsState>()(
  persist(
    (set) => ({
      filters: DEFAULT_EVENT_FILTERS,
      sort: 'time-asc',
      selectionMode: false,

      setFilters: (filters) => set({ filters }),
      setSort: (sort) => set({ sort }),
      setSelectionMode: (selectionMode) => set({ selectionMode }),
      resetFilters: () => set({ filters: DEFAULT_EVENT_FILTERS }),
    }),
    {
      name: 'trafficpulse-workspace-prefs',
      partialize: (state) => ({
        filters: state.filters,
        sort: state.sort,
        selectionMode: state.selectionMode,
      }),
    },
  ),
);
