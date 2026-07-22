import { useMemo, useRef } from 'react';

import { type ConfirmedEvent, type EvidenceManifest } from '@/api/types';
import { type WorkspaceEvent, mergeWorkspaceEvents, toWorkspaceEvent } from '@/lib/workspace';
import { useSelectionStore } from '@/store/selection-store';

import { useEvent, useEventsList, useEvidence } from './use-events';

const WORKSPACE_EVENT_LIMIT = 200;
/** Poll cadence for the event list while a job is processing (ms). */
export const EVENTS_POLL_INTERVAL_MS = 2000;

export interface UseWorkspaceEventsOptions {
  /** True while the job is processing → poll for newly-confirmed events (H7D). */
  active?: boolean;
}

export interface WorkspaceEventsResult {
  events: WorkspaceEvent[];
  total: number;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  refetch: () => void;
  selectedEventId: string | null;
  selectedEvent: WorkspaceEvent | null;
  selectedDetail: ConfirmedEvent | undefined;
  isDetailLoading: boolean;
  detailError: unknown;
  refetchDetail: () => void;
  evidence: EvidenceManifest | undefined;
  isEvidenceLoading: boolean;
  evidenceError: unknown;
  refetchEvidence: () => void;
  select: (eventId: string | null) => void;
}

/**
 * Workspace events (H7C, live in H7D): the video's confirmed events as
 * view-models, plus the selected event's detail + evidence.
 *
 * While `active`, the list is polled so events appear as the run confirms them,
 * and each poll is merged into the prior set ({@link mergeWorkspaceEvents}) to
 * preserve object identity — so appends update only what changed, existing rows
 * and the current selection survive background refreshes, and an unchanged poll
 * causes no rerender. Detail and evidence expose their own error + retry so a
 * delayed or failed manifest is recoverable without a manual page refresh.
 */
export function useWorkspaceEvents(
  videoId: string | undefined,
  options?: UseWorkspaceEventsOptions,
): WorkspaceEventsResult {
  const selectedEventId = useSelectionStore((s) => s.selectedEventId);
  const select = useSelectionStore((s) => s.selectEvent);

  const listQuery = useEventsList(
    { videoId, limit: WORKSPACE_EVENT_LIMIT, offset: 0, sort: 'trigger_at' },
    { refetchInterval: options?.active ? EVENTS_POLL_INTERVAL_MS : false },
  );

  const detailQuery = useEvent(selectedEventId ?? undefined);
  const evidenceQuery = useEvidence(selectedEventId ?? undefined);

  const summaries = useMemo(() => listQuery.data?.items ?? [], [listQuery.data]);

  const rawEvents = useMemo(
    () =>
      summaries.map((summary) =>
        summary.event_id === selectedEventId
          ? toWorkspaceEvent(summary, detailQuery.data)
          : toWorkspaceEvent(summary),
      ),
    [summaries, selectedEventId, detailQuery.data],
  );

  // Preserve references across polls so unchanged rows/markers do not rerender.
  const mergedRef = useRef<WorkspaceEvent[]>([]);
  const events = useMemo(() => {
    const merged = mergeWorkspaceEvents(mergedRef.current, rawEvents);
    mergedRef.current = merged;
    return merged;
  }, [rawEvents]);

  const selectedEvent = useMemo(
    () => events.find((event) => event.id === selectedEventId) ?? null,
    [events, selectedEventId],
  );

  return {
    events,
    total: listQuery.data?.total ?? 0,
    isLoading: Boolean(videoId) && listQuery.isLoading,
    isError: listQuery.isError,
    error: listQuery.error,
    refetch: () => void listQuery.refetch(),
    selectedEventId,
    selectedEvent,
    selectedDetail: detailQuery.data,
    isDetailLoading: Boolean(selectedEventId) && detailQuery.isLoading,
    detailError: detailQuery.isError ? detailQuery.error : null,
    refetchDetail: () => void detailQuery.refetch(),
    evidence: evidenceQuery.data,
    isEvidenceLoading: Boolean(selectedEventId) && evidenceQuery.isLoading,
    evidenceError: evidenceQuery.isError ? evidenceQuery.error : null,
    refetchEvidence: () => void evidenceQuery.refetch(),
    select,
  };
}
