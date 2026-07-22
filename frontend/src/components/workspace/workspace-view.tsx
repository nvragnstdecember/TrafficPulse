import { useEffect, useMemo, useRef, useState } from 'react';

import { type ViolationType } from '@/api/types';
import { PLAYER_SHORTCUTS, usePlayerShortcuts } from '@/hooks/use-player-shortcuts';
import { type ProcessingController } from '@/hooks/use-processing';
import { useWorkspaceEvents } from '@/hooks/use-workspace-events';
import { isActivePhase } from '@/lib/job';
import {
  type EventFilters,
  type WorkspaceSort,
  ALL_VIOLATION_TYPES,
  DEFAULT_EVENT_FILTERS,
  buildTimelineMarkers,
  filterWorkspaceEvents,
  sortWorkspaceEvents,
  timelineDuration,
} from '@/lib/workspace';
import { useProcessingStore } from '@/store/processing-store';

import { ErrorBanner } from '../common/error-banner';
import { EventDetail } from './event-detail';
import { EventList } from './event-list';
import { usePlayer } from './player-context';
import { ProcessingPanel } from './processing-panel';
import { Timeline } from './timeline';
import { VideoPlayer } from './video-player';

export interface WorkspaceViewProps {
  processing: ProcessingController;
  /** Object URL for local playback, or null when the file is not in this session. */
  objectUrl: string | null;
}

/**
 * The workspace (H7C, live in H7D): player + timeline + processing on the left,
 * the event list and the selected event's detail on the right.
 *
 * The composition layer. It holds only view state (filters, sort) and derives
 * everything else with the pure helpers in `lib/workspace`. In H7D it also:
 * polls events while the job is active (so markers appear as they arrive),
 * restores the selection and playback position after a refresh, mirrors both
 * back for the next recovery, and surfaces a reconnect banner when the job poll
 * is failing.
 */
export function WorkspaceView({ processing, objectUrl }: WorkspaceViewProps) {
  const { state, controls } = usePlayer();
  const [filters, setFilters] = useState<EventFilters>(DEFAULT_EVENT_FILTERS);
  const [sort, setSort] = useState<WorkspaceSort>('time-asc');

  const active = isActivePhase(processing.phase);
  const workspace = useWorkspaceEvents(processing.video?.video_id, { active });
  const { events, selectedEventId, selectedEvent, select } = workspace;

  // Route every selection through here so it is both applied and persisted for
  // recovery — no mirroring effect that could clobber the restored value.
  function handleSelect(eventId: string | null): void {
    select(eventId);
    useProcessingStore.getState().rememberSelection(eventId);
  }

  function selectAndSeek(eventId: string): void {
    handleSelect(eventId);
    const target = events.find((event) => event.id === eventId);
    if (target) controls.seek(target.mediaSeconds);
  }

  // Restore a persisted selection once, on mount (recovery after a refresh).
  const restoredSelectionRef = useRef(false);
  useEffect(() => {
    if (restoredSelectionRef.current) return;
    restoredSelectionRef.current = true;
    const persisted = useProcessingStore.getState().selectedEventId;
    if (persisted && !selectedEventId) select(persisted);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Restore the playback position once the media is ready, then mirror it.
  const restoredPlaybackRef = useRef(false);
  useEffect(() => {
    if (restoredPlaybackRef.current || !objectUrl) return;
    if (state.status === 'ready' || state.status === 'paused' || state.status === 'playing') {
      restoredPlaybackRef.current = true;
      const saved = useProcessingStore.getState().playbackSeconds;
      if (saved > 0) controls.seek(saved);
    }
  }, [state.status, objectUrl, controls]);

  useEffect(() => {
    useProcessingStore.getState().rememberPlayback(state.currentTime);
  }, [state.currentTime]);

  const visibleEvents = useMemo(
    () => sortWorkspaceEvents(filterWorkspaceEvents(events, filters), sort),
    [events, filters, sort],
  );

  const duration = timelineDuration(state.duration, events);
  const markers = useMemo(() => buildTimelineMarkers(events, duration), [events, duration]);

  const availableViolations = useMemo<ViolationType[]>(() => {
    const present = new Set(events.map((event) => event.violationType));
    return present.size > 0
      ? ALL_VIOLATION_TYPES.filter((type) => present.has(type))
      : ALL_VIOLATION_TYPES;
  }, [events]);

  function stepEvent(offset: 1 | -1): void {
    if (visibleEvents.length === 0) return;
    const index = visibleEvents.findIndex((event) => event.id === selectedEventId);
    const next =
      index < 0
        ? offset > 0
          ? 0
          : visibleEvents.length - 1
        : (index + offset + visibleEvents.length) % visibleEvents.length;
    selectAndSeek(visibleEvents[next].id);
  }

  usePlayerShortcuts({
    state,
    controls,
    onNextEvent: () => stepEvent(1),
    onPreviousEvent: () => stepEvent(-1),
  });

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div className="space-y-4">
        {processing.connectionError ? (
          <ErrorBanner
            title="Lost connection to the server"
            error={processing.connectionError}
            onRetry={processing.actions.reconnect}
          />
        ) : null}
        <VideoPlayer src={objectUrl} />
        <Timeline
          markers={markers}
          duration={duration}
          selectedEventId={selectedEventId}
          onSelect={handleSelect}
        />
        <ProcessingPanel controller={processing} />
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-muted-foreground">
          {PLAYER_SHORTCUTS.map((shortcut) => (
            <div key={shortcut.keys} className="flex items-center gap-1.5">
              <dt className="rounded border px-1 py-0.5 font-mono">{shortcut.keys}</dt>
              <dd>{shortcut.description}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="space-y-4">
        <EventList
          events={visibleEvents}
          totalCount={events.length}
          selectedEventId={selectedEventId}
          onSelect={selectAndSeek}
          filters={filters}
          onFiltersChange={setFilters}
          sort={sort}
          onSortChange={setSort}
          availableViolations={availableViolations}
          isLoading={workspace.isLoading}
          isError={workspace.isError}
          error={workspace.error}
          onRetry={workspace.refetch}
        />
        <EventDetail
          event={selectedEvent}
          detail={workspace.selectedDetail}
          evidence={workspace.evidence}
          isLoading={workspace.isDetailLoading}
          detailError={workspace.detailError}
          onRetryDetail={workspace.refetchDetail}
          isEvidenceLoading={workspace.isEvidenceLoading}
          evidenceError={workspace.evidenceError}
          onRetryEvidence={workspace.refetchEvidence}
          onSeek={controls.seek}
        />
      </div>
    </div>
  );
}
