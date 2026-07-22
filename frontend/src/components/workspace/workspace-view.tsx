import { useMemo, useState } from 'react';

import { type ViolationType } from '@/api/types';
import { type ProcessingController } from '@/hooks/use-processing';
import { PLAYER_SHORTCUTS, usePlayerShortcuts } from '@/hooks/use-player-shortcuts';
import { useWorkspaceEvents } from '@/hooks/use-workspace-events';
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
 * The workspace (H7C): player + timeline + processing on the left, the event list
 * and the selected event's detail on the right.
 *
 * This is the composition layer — it holds only view state (filters, sort) and
 * derives everything else with the pure helpers in `lib/workspace`. Data comes
 * from `useWorkspaceEvents`, the workflow from `useProcessing`, and playback from
 * the shared player controller.
 */
export function WorkspaceView({ processing, objectUrl }: WorkspaceViewProps) {
  const { state, controls } = usePlayer();
  const [filters, setFilters] = useState<EventFilters>(DEFAULT_EVENT_FILTERS);
  const [sort, setSort] = useState<WorkspaceSort>('time-asc');

  const workspace = useWorkspaceEvents(processing.video?.video_id);
  const { events, selectedEventId, selectedEvent, select } = workspace;

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

  /** Select an event and move the playhead to it — list, timeline, and player stay in sync. */
  function selectAndSeek(eventId: string): void {
    select(eventId);
    const target = events.find((event) => event.id === eventId);
    if (target) controls.seek(target.mediaSeconds);
  }

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
        <VideoPlayer src={objectUrl} />
        <Timeline
          markers={markers}
          duration={duration}
          selectedEventId={selectedEventId}
          onSelect={select}
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
          onSeek={controls.seek}
        />
      </div>
    </div>
  );
}
