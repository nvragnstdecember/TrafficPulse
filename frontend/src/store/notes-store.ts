import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface NotesState {
  /** Analyst notes keyed by event id (local only — never sent to the backend). */
  notes: Record<string, string>;
  setNote: (eventId: string, text: string) => void;
  clearNote: (eventId: string) => void;
}

/**
 * Analyst notes (H7E) — local, per-event free text.
 *
 * Persisted to `localStorage` so a reviewer's working notes survive a refresh,
 * but strictly client-side: the backend has no notes concept and none is sent.
 * Blank notes are pruned so the store never accumulates empty entries.
 */
export const useNotesStore = create<NotesState>()(
  persist(
    (set) => ({
      notes: {},
      setNote: (eventId, text) =>
        set((state) => {
          const next = { ...state.notes };
          if (text.trim().length === 0) delete next[eventId];
          else next[eventId] = text;
          return { notes: next };
        }),
      clearNote: (eventId) =>
        set((state) => {
          if (!(eventId in state.notes)) return state;
          const next = { ...state.notes };
          delete next[eventId];
          return { notes: next };
        }),
    }),
    { name: 'trafficpulse-notes' },
  ),
);

/** Read one event's note (empty string when none). */
export function useEventNote(eventId: string | null): string {
  return useNotesStore((state) => (eventId ? (state.notes[eventId] ?? '') : ''));
}
