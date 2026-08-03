import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import {
  type EventFilters,
  type WorkspaceSort,
  DEFAULT_EVENT_FILTERS,
  DEFAULT_WORKSPACE_SORT,
  normalizeEventFilters,
  normalizeWorkspaceSort,
} from '@/lib/workspace';

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
      sort: DEFAULT_WORKSPACE_SORT,
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
      // Rehydration is a trust boundary. The stored blob was written by
      // whichever version of the app the analyst last used, and `EventFilters`
      // has gained fields since this store shipped (H7E) — notably H9's
      // `reviewStatuses`. Zustand's default merge is shallow, so a stale
      // `filters` object replaces the defaults *wholesale* rather than filling
      // in around them, and the missing arrays then blow up on first render.
      //
      // Normalizing here (rather than versioning + migrating) makes that safe
      // by construction: `merge` runs on every rehydration, so a field added
      // later is defaulted automatically, with no version bump to forget.
      merge: (persisted, current) => {
        const raw = (
          typeof persisted === 'object' && persisted !== null ? persisted : {}
        ) as Partial<WorkspacePrefsState>;
        return {
          ...current,
          filters: normalizeEventFilters(raw.filters),
          sort: normalizeWorkspaceSort(raw.sort),
          selectionMode: raw.selectionMode === true,
        };
      },
    },
  ),
);
