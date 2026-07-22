import { create } from 'zustand';

interface SelectionState {
  /** The video the user is currently focused on across the app. */
  currentVideoId: string | null;
  /** The event selected in the workspace (drives player + timeline + details). */
  selectedEventId: string | null;
  selectVideo: (videoId: string | null) => void;
  clearSelection: () => void;
  selectEvent: (eventId: string | null) => void;
  clearEventSelection: () => void;
}

/** Transient cross-page selection state (H7B/H7C). Not persisted. */
export const useSelectionStore = create<SelectionState>((set) => ({
  currentVideoId: null,
  selectedEventId: null,
  selectVideo: (videoId) => set({ currentVideoId: videoId }),
  clearSelection: () => set({ currentVideoId: null, selectedEventId: null }),
  selectEvent: (eventId) => set({ selectedEventId: eventId }),
  clearEventSelection: () => set({ selectedEventId: null }),
}));
