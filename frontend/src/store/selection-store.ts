import { create } from 'zustand';

interface SelectionState {
  /** The video the user is currently focused on across the app. */
  currentVideoId: string | null;
  /** The event selected in the workspace (drives player + timeline + details). */
  selectedEventId: string | null;
  /**
   * Events checked for a bulk action (multi-select; H7E). Distinct from the
   * single active selection above, which drives playback and the detail panel.
   */
  checkedEventIds: Set<string>;
  selectVideo: (videoId: string | null) => void;
  clearSelection: () => void;
  selectEvent: (eventId: string | null) => void;
  clearEventSelection: () => void;
  toggleChecked: (eventId: string) => void;
  setChecked: (eventIds: string[]) => void;
  clearChecked: () => void;
}

/** Transient cross-page selection state (H7B/H7C/H7E). Not persisted. */
export const useSelectionStore = create<SelectionState>((set) => ({
  currentVideoId: null,
  selectedEventId: null,
  checkedEventIds: new Set<string>(),
  selectVideo: (videoId) => set({ currentVideoId: videoId }),
  clearSelection: () =>
    set({ currentVideoId: null, selectedEventId: null, checkedEventIds: new Set<string>() }),
  selectEvent: (eventId) => set({ selectedEventId: eventId }),
  clearEventSelection: () => set({ selectedEventId: null }),
  toggleChecked: (eventId) =>
    set((state) => {
      const next = new Set(state.checkedEventIds);
      if (next.has(eventId)) next.delete(eventId);
      else next.add(eventId);
      return { checkedEventIds: next };
    }),
  setChecked: (eventIds) => set({ checkedEventIds: new Set(eventIds) }),
  clearChecked: () => set({ checkedEventIds: new Set<string>() }),
}));
