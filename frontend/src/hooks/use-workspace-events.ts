import { useMemo } from 'react';

import { type ConfirmedEvent, type EvidenceManifest } from '@/api/types';
import { type WorkspaceEvent, toWorkspaceEvent } from '@/lib/workspace';
import { useSelectionStore } from '@/store/selection-store';

import { useEvent, useEventsList, useEvidence } from './use-events';

const WORKSPACE_EVENT_LIMIT = 200;

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
  evidence: EvidenceManifest | undefined;
  isDetailLoading: boolean;
  select: (eventId: string | null) => void;
}

/**
 * Workspace events (H7C): the video's confirmed events as view-models, plus the
 * selected event's detail + evidence. Composes the existing `useEventsList`,
 * `useEvent`, and `useEvidence` query hooks with the selection store.
 */
export function useWorkspaceEvents(videoId: string | undefined): WorkspaceEventsResult {
  const selectedEventId = useSelectionStore((s) => s.selectedEventId);
  const select = useSelectionStore((s) => s.selectEvent);

  const listQuery = useEventsList({
    videoId,
    limit: WORKSPACE_EVENT_LIMIT,
    offset: 0,
    sort: 'trigger_at',
  });

  const detailQuery = useEvent(selectedEventId ?? undefined);
  const evidenceQuery = useEvidence(selectedEventId ?? undefined);

  const summaries = useMemo(() => listQuery.data?.items ?? [], [listQuery.data]);

  const events = useMemo(
    () =>
      summaries.map((summary) =>
        summary.event_id === selectedEventId
          ? toWorkspaceEvent(summary, detailQuery.data)
          : toWorkspaceEvent(summary),
      ),
    [summaries, selectedEventId, detailQuery.data],
  );

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
    evidence: evidenceQuery.data,
    isDetailLoading: Boolean(selectedEventId) && detailQuery.isLoading,
    select,
  };
}
