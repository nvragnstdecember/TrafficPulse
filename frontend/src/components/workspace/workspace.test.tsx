import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { type ProcessingController } from '@/hooks/use-processing';
import { type EventFilters, DEFAULT_EVENT_FILTERS, buildTimelineMarkers } from '@/lib/workspace';
import {
  makeConfirmedEvent,
  makeEventSummary,
  makeEvidence,
  makeFile,
  makeJob,
  makeVideo,
  makeWorkspaceEvent,
  mediaSeconds,
} from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { ProgressBar } from '../common/progress-bar';
import { EventCard } from './event-card';
import { EventDetail } from './event-detail';
import { EventFiltersBar } from './event-filters';
import { EventList } from './event-list';
import { PlayerProvider } from './player-context';
import { ProcessingPanel } from './processing-panel';
import { Timeline } from './timeline';
import { UploadDropzone } from './upload-dropzone';
import { VideoPlayer } from './video-player';

function renderInPlayer(ui: React.ReactElement) {
  return renderWithProviders(<PlayerProvider fps={25}>{ui}</PlayerProvider>);
}

describe('ProgressBar', () => {
  it('exposes a determinate percentage', () => {
    renderWithProviders(<ProgressBar value={0.42} label="Upload progress" />);
    const bar = screen.getByRole('progressbar', { name: 'Upload progress' });
    expect(bar).toHaveAttribute('aria-valuenow', '42');
  });

  it('omits the value when indeterminate', () => {
    renderWithProviders(<ProgressBar value={null} label="Processing progress" />);
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
  });
});

describe('UploadDropzone', () => {
  it('accepts a valid file through the picker', async () => {
    const user = userEvent.setup();
    const onFileSelected = vi.fn();
    renderWithProviders(<UploadDropzone onFileSelected={onFileSelected} />);

    await user.upload(screen.getByTestId('upload-input'), makeFile('clip.mp4'));

    expect(onFileSelected).toHaveBeenCalledTimes(1);
    expect(onFileSelected.mock.calls[0][0].name).toBe('clip.mp4');
  });

  it('rejects an unsupported file inline, without notifying the workflow', async () => {
    const user = userEvent.setup();
    const onFileSelected = vi.fn();
    renderWithProviders(<UploadDropzone onFileSelected={onFileSelected} />);

    await user.upload(screen.getByTestId('upload-input'), makeFile('notes.txt'));

    expect(onFileSelected).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toHaveTextContent(/Unsupported format/);
  });

  it('shows the configured constraints and a workflow error', () => {
    renderWithProviders(<UploadDropzone onFileSelected={vi.fn()} error="Upload rejected" />);
    expect(screen.getByRole('button', { name: 'Upload a video' })).toHaveAccessibleDescription(
      /\.mp4/,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Upload rejected');
  });

  it('disables the zone while the workflow is busy', () => {
    renderWithProviders(<UploadDropzone onFileSelected={vi.fn()} disabled />);
    expect(screen.getByRole('button', { name: 'Upload a video' })).toBeDisabled();
  });

  it('accepts a dropped file', () => {
    const onFileSelected = vi.fn();
    renderWithProviders(<UploadDropzone onFileSelected={onFileSelected} />);
    const zone = screen.getByRole('button', { name: 'Upload a video' });

    fireEvent.dragOver(zone);
    fireEvent.drop(zone, { dataTransfer: { files: [makeFile('dropped.mp4')] } });

    expect(onFileSelected).toHaveBeenCalledTimes(1);
    expect(onFileSelected.mock.calls[0][0].name).toBe('dropped.mp4');
  });

  it('ignores a drop while disabled', () => {
    const onFileSelected = vi.fn();
    renderWithProviders(<UploadDropzone onFileSelected={onFileSelected} disabled />);
    const zone = screen.getByRole('button', { name: 'Upload a video' });

    fireEvent.drop(zone, { dataTransfer: { files: [makeFile('dropped.mp4')] } });

    expect(onFileSelected).not.toHaveBeenCalled();
  });
});

function makeController(overrides: Partial<ProcessingController> = {}): ProcessingController {
  return {
    phase: 'running',
    job: makeJob(),
    video: makeVideo(),
    progressRatio: 0.5,
    elapsedSeconds: 42,
    etaSeconds: 30,
    logs: [
      {
        id: 'log-1',
        at: Date.parse('2026-01-01T00:00:00Z'),
        level: 'info',
        message: 'Processing video…',
      },
    ],
    error: null,
    isBusy: true,
    isCancelling: false,
    connectionError: null,
    actions: {
      selectAndUpload: vi.fn(),
      startProcessing: vi.fn(),
      cancel: vi.fn(),
      cancelUpload: vi.fn(),
      retry: vi.fn(),
      remove: vi.fn(),
      replace: vi.fn(),
      reconnect: vi.fn(),
    },
    ...overrides,
  };
}

describe('ProcessingPanel', () => {
  it('renders the live phase, progress, and job statistics', () => {
    renderWithProviders(<ProcessingPanel controller={makeController()} />);

    expect(screen.getByText('junction.mp4')).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
    expect(screen.getByText('375 / 750')).toBeInTheDocument();
    expect(screen.getByText('12.5 fps')).toBeInTheDocument();
    expect(screen.getByText('Processing video…')).toBeInTheDocument();
  });

  it('offers cancel while uploading', async () => {
    const user = userEvent.setup();
    const controller = makeController({ phase: 'uploading', job: undefined, progressRatio: 0.2 });
    renderWithProviders(<ProcessingPanel controller={controller} />);

    await user.click(screen.getByRole('button', { name: /cancel upload/i }));
    expect(controller.actions.cancel).toHaveBeenCalled();
  });

  it('offers retry and shows the error when processing failed', async () => {
    const user = userEvent.setup();
    const controller = makeController({
      phase: 'failed',
      progressRatio: null,
      isBusy: false,
      error: 'decoder crashed',
    });
    renderWithProviders(<ProcessingPanel controller={controller} />);

    expect(screen.getByRole('alert')).toHaveTextContent('decoder crashed');
    await user.click(screen.getByRole('button', { name: /retry/i }));
    expect(controller.actions.retry).toHaveBeenCalled();
  });

  it('replaces the video from the picker once processing is idle', async () => {
    const user = userEvent.setup();
    const controller = makeController({ phase: 'completed', isBusy: false });
    renderWithProviders(<ProcessingPanel controller={controller} />);

    await user.upload(screen.getByTestId('replace-input'), makeFile('second.mp4'));
    expect(controller.actions.replace).toHaveBeenCalled();
  });

  it('confirms before removing the video', async () => {
    const user = userEvent.setup();
    const controller = makeController({ phase: 'completed', isBusy: false });
    renderWithProviders(<ProcessingPanel controller={controller} />);

    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    const dialog = await screen.findByRole('dialog');
    expect(controller.actions.remove).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole('button', { name: /^remove$/i }));
    expect(controller.actions.remove).toHaveBeenCalled();
  });

  it('cancels a running job through the unified cancel action (H7D)', async () => {
    const user = userEvent.setup();
    const controller = makeController({ phase: 'running' });
    renderWithProviders(<ProcessingPanel controller={controller} />);

    await user.click(screen.getByRole('button', { name: /^cancel$/i }));
    expect(controller.actions.cancel).toHaveBeenCalled();
  });

  it('shows the finalizing sub-phase (H7D)', () => {
    renderWithProviders(<ProcessingPanel controller={makeController({ phase: 'finalizing' })} />);
    expect(screen.getByText('Finalizing')).toBeInTheDocument();
  });

  it('disables cancel and shows progress while a cancellation is in flight (H7D)', () => {
    renderWithProviders(
      <ProcessingPanel controller={makeController({ phase: 'running', isCancelling: true })} />,
    );
    expect(screen.getByRole('button', { name: /cancelling/i })).toBeDisabled();
  });

  it('marks a cancelled job and offers a retry (H7D)', () => {
    renderWithProviders(
      <ProcessingPanel
        controller={makeController({ phase: 'cancelled', isBusy: false, progressRatio: 0.3 })}
      />,
    );
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });
});

describe('EventCard', () => {
  it('summarizes the violation and reports selection', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const event = makeWorkspaceEvent({ trigger_at: mediaSeconds(65) });
    renderWithProviders(<EventCard event={event} selected={false} onSelect={onSelect} />);

    const card = screen.getByRole('button', { name: 'Wrong way at 1:05' });
    expect(card).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByText('Conf —')).toBeInTheDocument();

    await user.click(card);
    expect(onSelect).toHaveBeenCalledWith(event.id);
  });
});

describe('EventFiltersBar', () => {
  function renderBar(filters: EventFilters = DEFAULT_EVENT_FILTERS) {
    const onFiltersChange = vi.fn();
    const onSortChange = vi.fn();
    renderWithProviders(
      <EventFiltersBar
        filters={filters}
        onFiltersChange={onFiltersChange}
        sort="time-asc"
        onSortChange={onSortChange}
        availableViolations={['wrong_way', 'no_helmet']}
      />,
    );
    return { onFiltersChange, onSortChange };
  }

  it('reports search text', async () => {
    const user = userEvent.setup();
    const { onFiltersChange } = renderBar();
    await user.type(screen.getByRole('searchbox'), 'c');
    expect(onFiltersChange).toHaveBeenCalledWith({ ...DEFAULT_EVENT_FILTERS, query: 'c' });
  });

  it('toggles a violation filter', async () => {
    const user = userEvent.setup();
    const { onFiltersChange } = renderBar();
    await user.click(screen.getByRole('button', { name: /violation/i }));
    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'No helmet' }));
    expect(onFiltersChange).toHaveBeenCalledWith({
      ...DEFAULT_EVENT_FILTERS,
      violationTypes: ['no_helmet'],
    });
  });

  it('changes the sort order', async () => {
    const user = userEvent.setup();
    const { onSortChange } = renderBar();
    await user.click(screen.getByRole('button', { name: 'Earliest first' }));
    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Latest first' }));
    expect(onSortChange).toHaveBeenCalledWith('time-desc');
  });

  it('clears active filters', async () => {
    const user = userEvent.setup();
    const { onFiltersChange } = renderBar({ ...DEFAULT_EVENT_FILTERS, query: 'cam' });
    await user.click(screen.getByRole('button', { name: 'Clear' }));
    expect(onFiltersChange).toHaveBeenCalledWith(DEFAULT_EVENT_FILTERS);
  });
});

describe('EventList', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(4) }),
    makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(30) }),
  ];

  const selectionProps = {
    selectionMode: false,
    onSelectionModeChange: vi.fn(),
    checkedIds: new Set<string>(),
    onToggleChecked: vi.fn(),
    onCheckAll: vi.fn(),
    onClearChecked: vi.fn(),
    onExport: vi.fn(),
  };

  function renderList(props: Partial<React.ComponentProps<typeof EventList>> = {}) {
    const onSelect = vi.fn();
    const onRetry = vi.fn();
    renderWithProviders(
      <EventList
        events={events}
        totalCount={events.length}
        selectedEventId="a"
        onSelect={onSelect}
        filters={DEFAULT_EVENT_FILTERS}
        onFiltersChange={vi.fn()}
        sort="time-asc"
        onSortChange={vi.fn()}
        availableViolations={['wrong_way']}
        isLoading={false}
        isError={false}
        onRetry={onRetry}
        {...selectionProps}
        {...props}
      />,
    );
    return { onSelect, onRetry };
  }

  it('lists the events with a filtered count', () => {
    renderList();
    expect(screen.getByText('2 of 2')).toBeInTheDocument();
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
  });

  it('shows skeletons while loading', () => {
    renderList({ isLoading: true });
    expect(screen.getByTestId('event-list-loading')).toBeInTheDocument();
  });

  it('offers a retry when the query failed', async () => {
    const user = userEvent.setup();
    const { onRetry } = renderList({ isError: true, error: new Error('boom') });
    await user.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalled();
  });

  it('distinguishes "no violations" from "no matches"', () => {
    const { unmount } = renderWithProviders(
      <EventList
        events={[]}
        totalCount={0}
        selectedEventId={null}
        onSelect={vi.fn()}
        filters={DEFAULT_EVENT_FILTERS}
        onFiltersChange={vi.fn()}
        sort="time-asc"
        onSortChange={vi.fn()}
        availableViolations={[]}
        isLoading={false}
        isError={false}
        onRetry={vi.fn()}
        {...selectionProps}
      />,
    );
    expect(screen.getByText('No violations detected')).toBeInTheDocument();
    unmount();

    renderList({ events: [], totalCount: 2 });
    expect(screen.getByText('No events match the filters')).toBeInTheDocument();
  });
});

describe('EventDetail', () => {
  const event = makeWorkspaceEvent({ trigger_at: mediaSeconds(65) });

  it('prompts for a selection when nothing is selected', () => {
    renderWithProviders(
      <EventDetail
        event={null}
        detail={undefined}
        evidence={undefined}
        isLoading={false}
        onSeek={vi.fn()}
      />,
    );
    expect(screen.getByText('No event selected')).toBeInTheDocument();
  });

  it('shows the overview and seeks to the event', async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={makeConfirmedEvent()}
        evidence={undefined}
        isLoading={false}
        onSeek={onSeek}
      />,
    );

    expect(screen.getByText('cam-north')).toBeInTheDocument();
    expect(screen.getByText('wrong-way-v1')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /jump to 1:05/i }));
    expect(onSeek).toHaveBeenCalledWith(65);
  });

  it('shows measurements against thresholds', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={makeConfirmedEvent()}
        evidence={undefined}
        isLoading={false}
        onSeek={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('tab', { name: 'Measurements' }));
    expect(await screen.findByText('heading_deviation_deg')).toBeInTheDocument();
    expect(screen.getByText('min_heading_deviation_deg')).toBeInTheDocument();
    expect(screen.getByText('172 deg')).toBeInTheDocument();
  });

  it('shows the evidence manifest as typed references', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={makeConfirmedEvent()}
        evidence={makeEvidence()}
        isLoading={false}
        onSeek={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('tab', { name: 'Evidence' }));
    expect(await screen.findByText('frames/evt-1-before.jpg')).toBeInTheDocument();
    expect(screen.getByText('Heading compared to legal direction')).toBeInTheDocument();
    expect(screen.getByText('rtdetr@1.0')).toBeInTheDocument();
  });

  it('shows a retryable "evidence unavailable" state (H7D)', async () => {
    const user = userEvent.setup();
    const onRetryEvidence = vi.fn();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={makeConfirmedEvent()}
        evidence={undefined}
        isLoading={false}
        evidenceError={new Error('Evidence not ready')}
        onRetryEvidence={onRetryEvidence}
        onSeek={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('tab', { name: 'Evidence' }));
    expect(await screen.findByText('Evidence unavailable')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetryEvidence).toHaveBeenCalled();
  });

  it('shows a retryable measurements error (H7D)', async () => {
    const user = userEvent.setup();
    const onRetryDetail = vi.fn();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={undefined}
        evidence={undefined}
        isLoading={false}
        detailError={new Error('detail down')}
        onRetryDetail={onRetryDetail}
        onSeek={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('tab', { name: 'Measurements' }));
    expect(await screen.findByText('Could not load measurements')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetryDetail).toHaveBeenCalled();
  });
});

describe('VideoPlayer', () => {
  it('explains when the local file is unavailable for playback', () => {
    renderInPlayer(<VideoPlayer src={null} />);
    expect(screen.getByText(/Playback isn’t available/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Play' })).not.toBeInTheDocument();
  });

  it('renders the media surface and transport controls for a local source', () => {
    renderInPlayer(<VideoPlayer src="blob:clip" />);
    expect(screen.getByLabelText('Video preview')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Seek' })).toBeDisabled();
    expect(screen.getByText('0:00 / 0:00')).toBeInTheDocument();
  });

  it('overlays a spinner while the media buffers', () => {
    renderInPlayer(<VideoPlayer src="blob:clip" />);
    fireEvent(screen.getByLabelText('Video preview'), new Event('waiting'));
    expect(screen.getByRole('status')).toHaveTextContent('Loading video');
  });

  it('overlays a recoverable error when the media fails', async () => {
    const user = userEvent.setup();
    renderInPlayer(<VideoPlayer src="blob:clip" />);
    const video = screen.getByLabelText('Video preview');

    fireEvent(video, new Event('error'));
    expect(screen.getByRole('alert')).toHaveTextContent('This video could not be played.');

    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('Timeline', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(10) }),
    makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(10.4) }),
    makeWorkspaceEvent({
      event_id: 'c',
      violation_type: 'no_helmet',
      trigger_at: mediaSeconds(80),
    }),
  ];
  const markers = buildTimelineMarkers(events, 100);

  it('clusters overlapping markers and labels them', () => {
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId={null} onSelect={vi.fn()} />,
    );
    expect(
      screen.getByRole('button', { name: /Wrong way at 0:10 and 1 more/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'No helmet at 1:20' })).toBeInTheDocument();
  });

  it('selects the event behind a marker', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId={null} onSelect={onSelect} />,
    );

    await user.click(screen.getByRole('button', { name: 'No helmet at 1:20' }));
    expect(onSelect).toHaveBeenCalledWith('c');
  });

  it('previews the hovered marker', async () => {
    const user = userEvent.setup();
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId={null} onSelect={vi.fn()} />,
    );

    await user.hover(screen.getByRole('button', { name: /Wrong way at 0:10 and 1 more/ }));
    expect(await screen.findByText('2 events')).toBeInTheDocument();
  });

  it('zooms the track in and out', async () => {
    const user = userEvent.setup();
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId={null} onSelect={vi.fn()} />,
    );

    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('2×')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeEnabled();
  });
});

describe('event summary fixtures', () => {
  it('anchors media time at the epoch, matching the backend contract', () => {
    expect(makeEventSummary({ trigger_at: mediaSeconds(3) }).trigger_at).toBe(
      '1970-01-01T00:00:03.000Z',
    );
  });
});
