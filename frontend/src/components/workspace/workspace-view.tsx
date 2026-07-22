import { useEffect, useMemo, useRef, useState } from 'react';

import { type ViolationType } from '@/api/types';
import { PLAYER_SHORTCUTS, usePlayerShortcuts } from '@/hooks/use-player-shortcuts';
import { type ProcessingController } from '@/hooks/use-processing';
import { useWorkspaceEvents } from '@/hooks/use-workspace-events';
import { isActivePhase } from '@/lib/job';
import {
  downloadTextFile,
  eventsToCsv,
  eventsToJsonModel,
  exportFilename,
  jsonString,
} from '@/lib/export';
import {
  ALL_VIOLATION_TYPES,
  type WorkspaceEvent,
  buildTimelineMarkers,
  filterWorkspaceEvents,
  sortWorkspaceEvents,
  timelineDuration,
} from '@/lib/workspace';
import { useProcessingStore } from '@/store/processing-store';
import { useSelectionStore } from '@/store/selection-store';
import { useWorkspacePrefsStore } from '@/store/workspace-prefs-store';
import { notify } from '@/store/notifications-store';

import { ErrorBanner } from '../common/error-banner';
import { type ExportFormat, EventList } from './event-list';
import { EventDetail } from './event-detail';
import { EvidenceViewer } from './evidence-viewer';
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
 * The workspace (H7C live in H7D, analyst review in H7E): player + timeline +
 * processing on the left; the event list, detail panel, and evidence viewer on
 * the right.
 *
 * The composition layer. Filters/sort come from the persisted prefs store (a
 * single source, remembered across refreshes); multi-select lives in the
 * selection store; export reuses already-fetched data through pure serializers.
 * Everything else is derived with the pure helpers in `lib/workspace`.
 */
export function WorkspaceView({ processing, objectUrl }: WorkspaceViewProps) {
  const { state, controls } = usePlayer();

  const filters = useWorkspacePrefsStore((s) => s.filters);
  const setFilters = useWorkspacePrefsStore((s) => s.setFilters);
  const sort = useWorkspacePrefsStore((s) => s.sort);
  const setSort = useWorkspacePrefsStore((s) => s.setSort);
  const selectionMode = useWorkspacePrefsStore((s) => s.selectionMode);
  const setSelectionMode = useWorkspacePrefsStore((s) => s.setSelectionMode);

  const checkedIds = useSelectionStore((s) => s.checkedEventIds);
  const toggleChecked = useSelectionStore((s) => s.toggleChecked);
  const setChecked = useSelectionStore((s) => s.setChecked);
  const clearChecked = useSelectionStore((s) => s.clearChecked);

  const [viewerOpen, setViewerOpen] = useState(false);

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

  // --- export (reuses already-fetched data; no duplicated serialization) ---
  const exportScope = (): WorkspaceEvent[] =>
    checkedIds.size > 0 ? events.filter((event) => checkedIds.has(event.id)) : visibleEvents;

  function handleExportList(format: ExportFormat): void {
    const scope = exportScope();
    if (scope.length === 0) return;
    const base = `trafficpulse-events-${scope.length}`;
    const ok =
      format === 'csv'
        ? downloadTextFile(exportFilename(base, 'csv'), eventsToCsv(scope), 'text/csv')
        : downloadTextFile(
            exportFilename(base, 'json'),
            jsonString(eventsToJsonModel(scope)),
            'application/json',
          );
    if (ok) notify({ title: `Exported ${scope.length} event(s) as ${format.toUpperCase()}.` });
  }

  function handleExportEventJson(): void {
    if (!selectedEvent) return;
    const data = workspace.selectedDetail ?? eventsToJsonModel([selectedEvent])[0];
    downloadTextFile(
      exportFilename(`event-${selectedEvent.id}`, 'json'),
      jsonString(data),
      'application/json',
    );
    notify({ title: 'Exported event JSON.' });
  }

  function handleExportManifest(): void {
    if (!selectedEvent || !workspace.evidence) return;
    downloadTextFile(
      exportFilename(`evidence-${selectedEvent.id}`, 'json'),
      jsonString(workspace.evidence),
      'application/json',
    );
    notify({ title: 'Exported evidence manifest.' });
  }

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
          selectionMode={selectionMode}
          onSelectionModeChange={setSelectionMode}
          checkedIds={checkedIds}
          onToggleChecked={toggleChecked}
          onCheckAll={() => setChecked(visibleEvents.map((event) => event.id))}
          onClearChecked={clearChecked}
          onExport={handleExportList}
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
          onOpenEvidenceViewer={() => setViewerOpen(true)}
          onExportJson={handleExportEventJson}
          onExportManifest={handleExportManifest}
        />
      </div>

      <EvidenceViewer
        open={viewerOpen}
        onOpenChange={setViewerOpen}
        event={selectedEvent}
        evidence={workspace.evidence}
        objectUrl={objectUrl}
        fps={processing.video?.fps}
      />
    </div>
  );
}
