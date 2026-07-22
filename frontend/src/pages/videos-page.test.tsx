import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { DEFAULT_EVENT_FILTERS } from '@/lib/workspace';
import { useNotesStore } from '@/store/notes-store';
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
  makeVideo,
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
  },
}));

vi.mock('@/services/events.service', () => ({
  eventsService: {
    list: vi.fn(),
    get: vi.fn(),
    getEvidence: vi.fn(),
  },
}));

const { videosService } = await import('@/services/videos.service');
const { eventsService } = await import('@/services/events.service');

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useUploadStore.getState().reset();
    useProcessingStore.getState().reset();
    useSelectionStore.getState().clearSelection();
    useNotesStore.setState({ notes: {} });
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
  vi.mocked(eventsService.get).mockResolvedValue(makeConfirmedEvent());
  vi.mocked(eventsService.getEvidence).mockResolvedValue(makeEvidence());
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

  it('saves an analyst note for the selected event', async () => {
    const user = await uploadAndOpenWorkspace();
    const list = screen.getByRole('region', { name: 'Detected events' });
    await user.click(await within(list).findByRole('button', { name: 'Wrong way at 0:04' }));

    const notes = await screen.findByPlaceholderText('Add a review note…');
    await user.type(notes, 'Confirmed');
    expect(useNotesStore.getState().notes['evt-1']).toBe('Confirmed');
  });
});
