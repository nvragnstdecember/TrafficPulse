import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { DEFAULT_EVENT_FILTERS } from '@/lib/workspace';
import { useNotificationsStore } from '@/store/notifications-store';
import { useProcessingStore } from '@/store/processing-store';
import { useSelectionStore } from '@/store/selection-store';
import { useUploadStore } from '@/store/upload-store';
import { useWorkspacePrefsStore } from '@/store/workspace-prefs-store';
import {
  makeConfirmedEvent,
  makeEventSummary,
  makeEvidence,
  makeFile,
  makeJob,
  makeReview,
  makeReviewCase,
  makeReviewEntry,
  makeSceneSummary,
  makeVideo,
  makeVideoSummary,
  mediaSeconds,
} from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import VideosPage from './videos-page';

vi.mock('@/services/videos.service', () => ({
  videosService: {
    upload: vi.fn(),
    startProcessing: vi.fn(),
    getJob: vi.fn(),
    cancelJob: vi.fn(),
    list: vi.fn(),
    getVideo: vi.fn(),
  },
}));

vi.mock('@/services/scenes.service', () => ({
  scenesService: {
    getForVideo: vi.fn(),
    calibrate: vi.fn(),
    validate: vi.fn(),
  },
}));

vi.mock('@/services/events.service', () => ({
  eventsService: {
    list: vi.fn(),
    get: vi.fn(),
    getEvidence: vi.fn(),
    getReview: vi.fn(),
    decide: vi.fn(),
  },
}));

const { videosService } = await import('@/services/videos.service');
const { eventsService } = await import('@/services/events.service');
const { scenesService } = await import('@/services/scenes.service');

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useUploadStore.getState().reset();
    useProcessingStore.getState().reset();
    useSelectionStore.getState().clearSelection();
    useWorkspacePrefsStore.setState({
      filters: DEFAULT_EVENT_FILTERS,
      sort: 'time-asc',
      selectionMode: false,
    });
  });
  vi.mocked(videosService.upload).mockResolvedValue(makeVideo());
  vi.mocked(videosService.startProcessing).mockResolvedValue({
    job_id: 'job-1',
    video_id: 'vid-1',
    status: 'pending',
  });
  vi.mocked(videosService.getJob).mockResolvedValue(
    makeJob({ status: 'succeeded', progress: 1, event_count: 2 }),
  );
  vi.mocked(videosService.cancelJob).mockResolvedValue(makeJob({ status: 'running' }));
  vi.mocked(videosService.list).mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0,
  });
  vi.mocked(eventsService.list).mockResolvedValue({
    items: [
      makeEventSummary({ event_id: 'evt-1', trigger_at: mediaSeconds(4) }),
      makeEventSummary({
        event_id: 'evt-2',
        violation_type: 'no_helmet',
        trigger_at: mediaSeconds(30),
      }),
    ],
    total: 2,
    limit: 200,
    offset: 0,
  });
  vi.mocked(scenesService.getForVideo).mockRejectedValue(
    new ApiError('no scene', { kind: 'http', status: 404, type: 'scene_not_found' }),
  );
  vi.mocked(scenesService.calibrate).mockResolvedValue(makeSceneSummary());
  vi.mocked(scenesService.validate).mockResolvedValue({
    valid: true,
    errors: [],
    supported_violations: ['wrong_way'],
    scene_hash: 'abc',
  });
  vi.mocked(eventsService.get).mockResolvedValue(makeConfirmedEvent());
  vi.mocked(eventsService.getEvidence).mockResolvedValue(makeEvidence());
  vi.mocked(eventsService.getReview).mockResolvedValue(makeReview());
  vi.mocked(eventsService.decide).mockResolvedValue(
    makeReview({
      case: makeReviewCase({ status: 'in_review', reviewer_id: 'analyst' }),
      history: [makeReviewEntry()],
    }),
  );
});

/** Upload a file and wait for the workspace to replace the dropzone. */
async function uploadAndOpenWorkspace() {
  const user = userEvent.setup();
  renderWithProviders(<VideosPage />);
  await user.upload(screen.getByTestId('upload-input'), makeFile('clip.mp4'));
  await screen.findByRole('region', { name: 'Detected events' });
  return user;
}

describe('VideosPage (video workspace)', () => {
  it('starts at the upload stage and queries no events yet', () => {
    renderWithProviders(<VideosPage />);

    expect(screen.getByRole('heading', { name: 'Video workspace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Upload a video' })).toBeInTheDocument();
    // Demo readiness (H8): a first-run "how it works" hint accompanies the dropzone.
    expect(screen.getByRole('region', { name: 'How it works' })).toBeInTheDocument();
    expect(eventsService.list).not.toHaveBeenCalled();
  });

  it('runs upload → processing → review after a file is chosen', async () => {
    await uploadAndOpenWorkspace();

    expect(videosService.upload).toHaveBeenCalledTimes(1);
    expect(videosService.startProcessing).toHaveBeenCalledWith({ videoId: 'vid-1' });
    expect(screen.getByText('junction.mp4')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Completed')).toBeInTheDocument());

    // Player, timeline, and the event list are all mounted for review.
    expect(screen.getByRole('region', { name: 'Timeline' })).toBeInTheDocument();
    expect(await screen.findByText('2 of 2')).toBeInTheDocument();
  });

  it('walks upload → processing → review → playback → evidence → back', async () => {
    // The Phase 2 acceptance path, executed rather than described: every stage of
    // the analyst workflow has to be reachable in one continuous session without
    // the page unmounting the player or losing the selection.
    const user = await uploadAndOpenWorkspace();

    // Processing settles, and the workflow stepper says where we are.
    await waitFor(() => expect(screen.getByText('Completed')).toBeInTheDocument());
    const nav = screen.getByRole('navigation', { name: 'Review workflow' });
    expect(within(nav).getByText('Review')).toBeInTheDocument();

    // The run summary reports what actually happened.
    expect(screen.getByText('Processing complete')).toBeInTheDocument();
    expect(screen.getByText('Violations confirmed')).toBeInTheDocument();

    // Review: pick a violation from the dashboard.
    // Scoped to the list: the timeline renders a marker with the same label.
    const list = screen.getByRole('region', { name: 'Detected events' });
    const card = await within(list).findByRole('button', { name: /No helmet at 0:30/ });
    await user.click(card);
    expect(card).toHaveAttribute('aria-pressed', 'true');

    // Evidence: the detail panel opens on the selected event and tells its story.
    await screen.findByText('evt-2');
    await user.click(screen.getByRole('tab', { name: 'Timeline' }));
    expect(await screen.findByText('Violation confirmed')).toBeInTheDocument();

    // Back to the dashboard: the list is still there with the selection intact.
    await user.click(screen.getByRole('tab', { name: 'Overview' }));
    expect(screen.getByRole('region', { name: 'Detected events' })).toBeInTheDocument();
    expect(useSelectionStore.getState().selectedEventId).toBe('evt-2');
  });

  it('searches by a clock timestamp read off the video', async () => {
    const user = await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    await user.type(screen.getByRole('searchbox'), '0:30');

    await waitFor(() => expect(screen.getByText('1 of 2')).toBeInTheDocument());
  });

  it('loads the detail of an event selected from the list', async () => {
    const user = await uploadAndOpenWorkspace();

    const list = screen.getByRole('region', { name: 'Detected events' });
    await user.click(await within(list).findByRole('button', { name: 'Wrong way at 0:04' }));

    await waitFor(() => expect(eventsService.get).toHaveBeenCalledWith('evt-1', expect.anything()));
    expect(await screen.findByText('wrong-way-v1')).toBeInTheDocument();
    expect(useSelectionStore.getState().selectedEventId).toBe('evt-1');
  });

  it('steps through events with the keyboard shortcuts', async () => {
    await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'l', bubbles: true }));
    });
    await waitFor(() => expect(useSelectionStore.getState().selectedEventId).toBe('evt-1'));

    act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'l', bubbles: true }));
    });
    await waitFor(() => expect(useSelectionStore.getState().selectedEventId).toBe('evt-2'));
  });

  it('filters the event list without refetching', async () => {
    const user = await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    await user.type(screen.getByRole('searchbox'), 'helmet');

    expect(await screen.findByText('1 of 2')).toBeInTheDocument();
  });

  it('returns to the upload stage when the video is removed', async () => {
    const user = await uploadAndOpenWorkspace();

    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    const dialog = await screen.findByRole('dialog');
    await user.click(within(dialog).getByRole('button', { name: /^remove$/i }));

    expect(await screen.findByRole('button', { name: 'Upload a video' })).toBeInTheDocument();
  });
});

describe('VideosPage — live processing (H7D)', () => {
  it('cancels a running job and reflects the cancelled state', async () => {
    const user = userEvent.setup();
    // A job that stays running until it is cancelled, then reports cancelled.
    vi.mocked(videosService.getJob).mockResolvedValue(makeJob({ status: 'running' }));
    renderWithProviders(<VideosPage />);
    await user.upload(screen.getByTestId('upload-input'), makeFile('clip.mp4'));
    await screen.findByRole('region', { name: 'Detected events' });
    await waitFor(() => expect(screen.getByText('Running')).toBeInTheDocument());

    // From cancel onward the backend reports the job cancelled.
    vi.mocked(videosService.getJob).mockResolvedValue(makeJob({ status: 'cancelled' }));
    await user.click(screen.getByRole('button', { name: /^cancel$/i }));

    expect(videosService.cancelJob).toHaveBeenCalledWith('job-1');
    expect(await screen.findByText('Cancelled')).toBeInTheDocument();
  });

  it('surfaces a reconnect banner when the job poll fails, then recovers', async () => {
    const user = userEvent.setup();
    vi.mocked(videosService.getJob).mockRejectedValue(
      new ApiError('Network request failed', { kind: 'network' }),
    );
    // Seed an in-flight job (as a page refresh into a running job would), then mount.
    act(() => useProcessingStore.getState().attachJob('vid-1', 'job-1'));
    act(() => useUploadStore.getState().markUploaded(makeVideo()));
    renderWithProviders(<VideosPage />);

    expect(await screen.findByText('Lost connection to the server')).toBeInTheDocument();

    // Recovery: the backend comes back and a reconnect re-polls successfully.
    vi.mocked(videosService.getJob).mockResolvedValue(
      makeJob({ status: 'succeeded', progress: 1, event_count: 2 }),
    );
    await user.click(screen.getByRole('button', { name: /retry/i }));
    await waitFor(() =>
      expect(screen.queryByText('Lost connection to the server')).not.toBeInTheDocument(),
    );
  });

  it('opens the existing video when an upload is a duplicate', async () => {
    const user = userEvent.setup();
    vi.mocked(videosService.upload).mockRejectedValue(
      new ApiError('an identical video already exists as vid-existing', {
        kind: 'http',
        status: 409,
        type: 'duplicate_video',
        videoId: 'vid-existing',
      }),
    );
    renderWithProviders(<VideosPage />);

    await user.upload(screen.getByTestId('upload-input'), makeFile('dupe.mp4'));

    // The workspace opens for the already-uploaded video and starts processing it.
    await screen.findByRole('region', { name: 'Detected events' });
    expect(videosService.startProcessing).toHaveBeenCalledWith({ videoId: 'vid-existing' });
  });

  it('restores the persisted selection after a refresh into a completed job', async () => {
    // Simulate a reload: the persisted store already knows the video, job, and
    // which event was selected.
    act(() => {
      useUploadStore.getState().markUploaded(makeVideo());
      useProcessingStore.getState().attachJob('vid-1', 'job-1');
      useProcessingStore.getState().rememberSelection('evt-2');
    });
    renderWithProviders(<VideosPage />);

    await screen.findByRole('region', { name: 'Detected events' });
    // The previously-selected event's detail is fetched on restore.
    await waitFor(() => expect(useSelectionStore.getState().selectedEventId).toBe('evt-2'));
  });
});

describe('VideosPage — historical video library (H11)', () => {
  it('lists stored videos beside the dropzone', async () => {
    vi.mocked(videosService.list).mockResolvedValue({
      items: [
        makeVideoSummary({ video_id: 'vid-old', filename: 'yesterday.mp4', event_count: 3 }),
        makeVideoSummary({ video_id: 'vid-older', filename: 'monday.mp4', event_count: 0 }),
      ],
      total: 2,
      limit: 50,
      offset: 0,
    });
    renderWithProviders(<VideosPage />);

    const library = await screen.findByRole('list', { name: 'Stored videos' });
    expect(within(library).getByText('yesterday.mp4')).toBeInTheDocument();
    expect(within(library).getByText('monday.mp4')).toBeInTheDocument();
    expect(within(library).getByText('3 events')).toBeInTheDocument();
    // A succeeded run that confirmed nothing is not the same as an unprocessed video.
    expect(within(library).getByText('No violations')).toBeInTheDocument();
  });

  it('opens a previously processed video without re-uploading it', async () => {
    // The milestone's success path: browse → select → the analysis loads, with no
    // upload call and no new job.
    const user = userEvent.setup();
    vi.mocked(videosService.list).mockResolvedValue({
      items: [makeVideoSummary({ video_id: 'vid-old', filename: 'yesterday.mp4' })],
      total: 1,
      limit: 50,
      offset: 0,
    });
    vi.mocked(videosService.getJob).mockResolvedValue(
      makeJob({ job_id: 'job-1', video_id: 'vid-old', status: 'succeeded', event_count: 2 }),
    );
    renderWithProviders(<VideosPage />);

    await user.click(await screen.findByRole('button', { name: 'Open yesterday.mp4' }));

    // The workspace mounts on the stored video, exactly as it would after an upload.
    await screen.findByRole('region', { name: 'Detected events' });
    expect(await screen.findByText('2 of 2')).toBeInTheDocument();
    expect(videosService.upload).not.toHaveBeenCalled();
    expect(videosService.startProcessing).not.toHaveBeenCalled();
    expect(useProcessingStore.getState().jobId).toBe('job-1');
  });

  it('plays a stored video from the server when this session has no local file', async () => {
    // The gap that made the library unusable on its own: playback was a browser
    // object URL for the picked file, which a reopened video never has.
    const user = userEvent.setup();
    vi.mocked(videosService.list).mockResolvedValue({
      items: [makeVideoSummary({ video_id: 'vid-old', filename: 'yesterday.mp4' })],
      total: 1,
      limit: 50,
      offset: 0,
    });
    renderWithProviders(<VideosPage />);

    await user.click(await screen.findByRole('button', { name: 'Open yesterday.mp4' }));

    const player = await screen.findByLabelText('Video preview');
    await waitFor(() => expect(player).toHaveAttribute('src', '/api/videos/vid-old/media'));
  });

  it('processes a stored video that has never been analysed', async () => {
    const user = userEvent.setup();
    vi.mocked(videosService.list).mockResolvedValue({
      items: [
        makeVideoSummary({
          video_id: 'vid-raw',
          filename: 'unprocessed.mp4',
          job_id: null,
          status: null,
          job_count: 0,
          event_count: 0,
        }),
      ],
      total: 1,
      limit: 50,
      offset: 0,
    });
    renderWithProviders(<VideosPage />);

    await user.click(await screen.findByRole('button', { name: 'Process unprocessed.mp4' }));

    await waitFor(() =>
      expect(videosService.startProcessing).toHaveBeenCalledWith({ videoId: 'vid-raw' }),
    );
  });

  it('shows a useful empty state for a repository with no videos', async () => {
    renderWithProviders(<VideosPage />);

    expect(await screen.findByText('No videos yet')).toBeInTheDocument();
    expect(screen.queryByRole('list', { name: 'Stored videos' })).not.toBeInTheDocument();
  });

  it('surfaces a retryable error when the library cannot be loaded', async () => {
    const user = userEvent.setup();
    vi.mocked(videosService.list).mockRejectedValue(
      new ApiError('Network request failed', { kind: 'network' }),
    );
    renderWithProviders(<VideosPage />);

    expect(await screen.findByText('Could not load the video library')).toBeInTheDocument();

    vi.mocked(videosService.list).mockResolvedValue({
      items: [makeVideoSummary()],
      total: 1,
      limit: 50,
      offset: 0,
    });
    await user.click(screen.getByRole('button', { name: /retry/i }));

    expect(await screen.findByRole('list', { name: 'Stored videos' })).toBeInTheDocument();
  });

  it('returns to the library from an open video', async () => {
    const user = await uploadAndOpenWorkspace();

    await user.click(screen.getByRole('button', { name: 'Video library' }));

    expect(await screen.findByRole('button', { name: 'Upload a video' })).toBeInTheDocument();
    expect(screen.getByText('No videos yet')).toBeInTheDocument();
  });
});

describe('VideosPage — scene calibration and red-light timing (H12/H13)', () => {
  it('offers calibration inside the workspace without leaving it', async () => {
    await uploadAndOpenWorkspace();

    expect(await screen.findByText('Scene calibration')).toBeInTheDocument();
    expect(screen.getByTestId('calibration-surface')).toBeInTheDocument();
    // The workspace is not replaced: the player and the event list stay mounted.
    expect(screen.getByRole('region', { name: 'Detected events' })).toBeInTheDocument();
  });

  it('shows the signal schedule only once a junction scene is calibrated', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['red_light_jumping'] }),
    );
    await uploadAndOpenWorkspace();

    expect(await screen.findByLabelText('Signal schedule')).toBeInTheDocument();
  });

  it('re-runs the analysis with the rules the calibrated scene unlocked', async () => {
    // The H13 success path through the page: calibrate, declare the timing, re-run.
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['wrong_way', 'red_light_jumping'] }),
    );
    const user = await uploadAndOpenWorkspace();
    await screen.findByLabelText('Signal schedule');

    await user.click(screen.getByRole('button', { name: /add phase/i }));
    vi.mocked(videosService.startProcessing).mockClear();
    await user.click(screen.getByRole('button', { name: /re-run analysis/i }));

    await waitFor(() => expect(videosService.startProcessing).toHaveBeenCalled());
    const [input] = vi.mocked(videosService.startProcessing).mock.calls[0];
    expect(input.rules).toEqual([
      { kind: 'wrong_way' },
      { kind: 'red_light_jumping', schedule: [{ at_seconds: 0, state: 'red' }] },
    ]);
  });

  it('omits red-light from the re-run when no signal timing was entered', async () => {
    // The backend refuses a schedule-less red-light rule; the client must not send
    // one and turn a knowable state into a failed run.
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['wrong_way', 'red_light_jumping'] }),
    );
    const user = await uploadAndOpenWorkspace();

    vi.mocked(videosService.startProcessing).mockClear();
    await user.click(await screen.findByRole('button', { name: /re-run analysis/i }));

    await waitFor(() => expect(videosService.startProcessing).toHaveBeenCalled());
    const [input] = vi.mocked(videosService.startProcessing).mock.calls[0];
    expect(input.rules).toEqual([{ kind: 'wrong_way' }]);
  });
});

describe('VideosPage — review workflow (H7E)', () => {
  it('exports the shown events to a downloaded file', async () => {
    const createObjectURL = vi.fn(() => 'blob:export');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL });
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    const user = await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    await user.click(screen.getByRole('button', { name: 'Export' }));
    await user.click(await screen.findByRole('menuitem', { name: 'CSV' }));

    expect(createObjectURL).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    await waitFor(() =>
      expect(
        useNotificationsStore
          .getState()
          .notifications.some((n) => /Exported 2 event/.test(n.title)),
      ).toBe(true),
    );

    clickSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it('opens the evidence viewer for the selected event', async () => {
    const user = await uploadAndOpenWorkspace();
    const list = screen.getByRole('region', { name: 'Detected events' });
    await user.click(await within(list).findByRole('button', { name: 'Wrong way at 0:04' }));

    await user.click(await screen.findByRole('button', { name: /evidence viewer/i }));
    expect(await screen.findByRole('dialog')).toHaveTextContent(/Evidence — Wrong way/);
  });

  it('bulk-selects events and reflects the count', async () => {
    const user = await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    await user.click(screen.getByRole('button', { name: 'Select' }));
    await user.click(await screen.findByRole('checkbox', { name: 'Select all events' }));

    expect(screen.getByText('2 selected')).toBeInTheDocument();
    expect(useSelectionStore.getState().checkedEventIds.size).toBe(2);
  });

  it('remembers filters in the persisted prefs store', async () => {
    const user = await uploadAndOpenWorkspace();
    await screen.findByText('2 of 2');

    await user.type(screen.getByRole('searchbox'), 'helmet');

    await waitFor(() => expect(useWorkspacePrefsStore.getState().filters.query).toBe('helmet'));
    const persisted = JSON.parse(localStorage.getItem('trafficpulse-workspace-prefs') ?? '{}');
    expect(persisted.state.filters.query).toBe('helmet');
  });

  it('records a decision with its note, and shows the audit history', async () => {
    // The H9 success path, end to end through the real components: open a case,
    // type a justification, approve, and see both the status and the history move.
    const user = await uploadAndOpenWorkspace();
    const list = screen.getByRole('region', { name: 'Detected events' });
    await user.click(await within(list).findByRole('button', { name: 'Wrong way at 0:04' }));

    await user.click(await screen.findByRole('tab', { name: 'Review' }));
    await user.click(await screen.findByRole('button', { name: 'Start review' }));

    vi.mocked(eventsService.decide).mockResolvedValueOnce(
      makeReview({
        case: makeReviewCase({ status: 'approved', reviewer_id: 'analyst', note: 'Confirmed' }),
        history: [
          makeReviewEntry(),
          makeReviewEntry({
            entry_id: 'rev-2',
            action: 'approve',
            status_before: 'in_review',
            status_after: 'approved',
            note: 'Confirmed',
          }),
        ],
      }),
    );
    await user.type(await screen.findByLabelText('Analyst note'), 'Confirmed');
    await user.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() =>
      expect(eventsService.decide).toHaveBeenCalledWith('evt-1', {
        action: 'approve',
        reviewer: 'analyst',
        note: 'Confirmed',
      }),
    );
    // The decision propagates to every surface at once — header badge, panel
    // status, metadata, and history — so scope the assertions.
    const audit = await screen.findByRole('region', { name: 'Audit history' });
    expect(within(audit).getByText('Review opened')).toBeInTheDocument();
    expect(within(audit).getByText('Approved')).toBeInTheDocument();
    expect(screen.getAllByText('Approved').length).toBeGreaterThan(1);
  });

  it('offers only the decisions legal from the current state', async () => {
    // A pending case cannot be approved: the lifecycle exists to make "somebody
    // looked at this" auditable, so the button is simply not offered.
    const user = await uploadAndOpenWorkspace();
    const list = screen.getByRole('region', { name: 'Detected events' });
    await user.click(await within(list).findByRole('button', { name: 'Wrong way at 0:04' }));
    await user.click(await screen.findByRole('tab', { name: 'Review' }));

    expect(await screen.findByRole('button', { name: 'Start review' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Approve' })).not.toBeInTheDocument();
  });
});
