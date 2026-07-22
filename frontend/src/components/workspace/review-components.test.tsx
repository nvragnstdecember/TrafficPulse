import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useNotesStore } from '@/store/notes-store';
import {
  makeConfirmedEvent,
  makeEvidence,
  makeWorkspaceEvent,
  mediaSeconds,
} from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { CollapsibleSection } from '../common/collapsible-section';
import { CopyButton } from '../common/copy-button';
import { EventCard } from './event-card';
import { type ExportFormat, EventList } from './event-list';
import { EventDetail } from './event-detail';
import { EvidenceViewer } from './evidence-viewer';
import { PlayerProvider } from './player-context';
import { Timeline } from './timeline';

function renderInPlayer(ui: React.ReactElement) {
  return renderWithProviders(<PlayerProvider fps={25}>{ui}</PlayerProvider>);
}

beforeEach(() => {
  localStorage.clear();
  useNotesStore.setState({ notes: {} });
});

describe('CopyButton', () => {
  it('copies the value and shows transient success', async () => {
    const user = userEvent.setup();
    renderWithProviders(<CopyButton value="evt-1" label="Copy event ID" />);

    await user.click(screen.getByRole('button', { name: 'Copy event ID' }));
    await expect(navigator.clipboard.readText()).resolves.toBe('evt-1');
    expect(await screen.findByRole('button', { name: /copied/i })).toBeInTheDocument();
  });
});

describe('CollapsibleSection', () => {
  it('toggles its region', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <CollapsibleSection title="Technical metadata" defaultOpen={false}>
        <p>hidden detail</p>
      </CollapsibleSection>,
    );
    const trigger = screen.getByRole('button', { name: 'Technical metadata' });
    expect(trigger).toHaveAttribute('aria-expanded', 'false');

    await user.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('hidden detail')).toBeVisible();
  });
});

describe('EventCard (H7E)', () => {
  it('shows a severity badge for the violation', () => {
    renderWithProviders(
      <EventCard
        event={makeWorkspaceEvent({ violation_type: 'no_helmet' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText('High')).toBeInTheDocument();
  });

  it('exposes a checkbox in selection mode', async () => {
    const user = userEvent.setup();
    const onToggleChecked = vi.fn();
    renderWithProviders(
      <EventCard
        event={makeWorkspaceEvent({ event_id: 'a' })}
        selected={false}
        onSelect={vi.fn()}
        showCheckbox
        checked={false}
        onToggleChecked={onToggleChecked}
      />,
    );
    await user.click(screen.getByRole('checkbox'));
    expect(onToggleChecked).toHaveBeenCalledWith('a');
  });
});

describe('EventList (H7E review tools)', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(4) }),
    makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(30) }),
  ];

  function renderList(overrides: Partial<React.ComponentProps<typeof EventList>> = {}) {
    const props = {
      events,
      totalCount: events.length,
      selectedEventId: 'a' as string | null,
      onSelect: vi.fn(),
      filters: { query: '', violationTypes: [], minConfidence: 0 },
      onFiltersChange: vi.fn(),
      sort: 'time-asc' as const,
      onSortChange: vi.fn(),
      availableViolations: ['wrong_way' as const],
      isLoading: false,
      isError: false,
      onRetry: vi.fn(),
      selectionMode: false,
      onSelectionModeChange: vi.fn(),
      checkedIds: new Set<string>(),
      onToggleChecked: vi.fn(),
      onCheckAll: vi.fn(),
      onClearChecked: vi.fn(),
      onExport: vi.fn() as (format: ExportFormat) => void,
      ...overrides,
    };
    renderWithProviders(<EventList {...props} />);
    return props;
  }

  it('toggles selection mode', async () => {
    const user = userEvent.setup();
    const props = renderList();
    await user.click(screen.getByRole('button', { name: 'Select' }));
    expect(props.onSelectionModeChange).toHaveBeenCalledWith(true);
  });

  it('shows a bulk bar and selects all in selection mode', async () => {
    const user = userEvent.setup();
    const props = renderList({ selectionMode: true });
    await user.click(screen.getByRole('checkbox', { name: 'Select all events' }));
    expect(props.onCheckAll).toHaveBeenCalled();
  });

  it('exports via the menu', async () => {
    const user = userEvent.setup();
    const props = renderList();
    await user.click(screen.getByRole('button', { name: 'Export' }));
    await user.click(await screen.findByRole('menuitem', { name: 'CSV' }));
    expect(props.onExport).toHaveBeenCalledWith('csv');
  });

  it('moves the active event with the arrow keys', () => {
    const props = renderList();
    fireEvent.keyDown(screen.getByRole('list', { name: 'Event results' }), { key: 'ArrowDown' });
    expect(props.onSelect).toHaveBeenCalledWith('b');
    fireEvent.keyDown(screen.getByRole('list', { name: 'Event results' }), { key: 'End' });
    expect(props.onSelect).toHaveBeenCalledWith('b');
  });
});

describe('EvidenceViewer (H7E)', () => {
  const event = makeWorkspaceEvent({ trigger_at: mediaSeconds(20) });

  it('renders the frames and metadata over the local media', () => {
    renderWithProviders(
      <EvidenceViewer
        open
        onOpenChange={vi.fn()}
        event={event}
        evidence={makeEvidence()}
        objectUrl="blob:clip"
        fps={25}
      />,
    );
    expect(screen.getByRole('group', { name: 'Evidence frames' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Trigger' })).toBeInTheDocument();
    expect(screen.getByLabelText(/Evidence frame: Trigger/)).toBeInTheDocument();
    // Manifest metadata is shown, with copy affordances.
    expect(screen.getByText('frames/evt-1-before.jpg')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy evidence ID' })).toBeInTheDocument();
  });

  it('zooms in and resets', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EvidenceViewer
        open
        onOpenChange={vi.fn()}
        event={event}
        evidence={makeEvidence()}
        objectUrl="blob:clip"
      />,
    );
    expect(screen.getByText('1.0×')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('1.5×')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Reset view' }));
    expect(screen.getByText('1.0×')).toBeInTheDocument();
  });

  it('switches frames', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EvidenceViewer
        open
        onOpenChange={vi.fn()}
        event={event}
        evidence={makeEvidence()}
        objectUrl="blob:clip"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Before' }));
    expect(screen.getByRole('button', { name: 'Before' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('explains when the local media is unavailable but still shows metadata', () => {
    renderWithProviders(
      <EvidenceViewer
        open
        onOpenChange={vi.fn()}
        event={event}
        evidence={undefined}
        objectUrl={null}
      />,
    );
    expect(screen.getByText(/Re-select the local video file/)).toBeInTheDocument();
    expect(screen.getByText(/manifest is not available/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
  });
});

describe('EventDetail (H7E)', () => {
  const event = makeWorkspaceEvent({ trigger_at: mediaSeconds(65) });

  it('copies the event id, visualizes confidence, and offers quick actions', async () => {
    const user = userEvent.setup();
    const onOpenEvidenceViewer = vi.fn();
    const onExportJson = vi.fn();
    renderWithProviders(
      <EventDetail
        event={{ ...event, confidence: 0.8 }}
        detail={makeConfirmedEvent()}
        evidence={makeEvidence()}
        isLoading={false}
        onSeek={vi.fn()}
        onOpenEvidenceViewer={onOpenEvidenceViewer}
        onExportJson={onExportJson}
        onExportManifest={vi.fn()}
      />,
    );

    expect(screen.getByRole('progressbar', { name: 'Event confidence' })).toHaveAttribute(
      'aria-valuenow',
      '80',
    );
    await user.click(screen.getByRole('button', { name: 'Copy event ID' }));
    await expect(navigator.clipboard.readText()).resolves.toBe(event.id);

    await user.click(screen.getByRole('button', { name: /evidence viewer/i }));
    expect(onOpenEvidenceViewer).toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /^json$/i }));
    expect(onExportJson).toHaveBeenCalled();
  });

  it('persists an analyst note to the local store', async () => {
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
    const notes = screen.getByPlaceholderText('Add a review note…');
    await user.type(notes, 'Reviewed');
    expect(useNotesStore.getState().notes[event.id]).toBe('Reviewed');
  });
});

describe('Timeline (H7E navigation)', () => {
  const markers = [
    {
      id: 'a',
      time: 10,
      positionRatio: 0.1,
      violationType: 'wrong_way' as const,
      event: makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(10) }),
    },
    {
      id: 'b',
      time: 80,
      positionRatio: 0.8,
      violationType: 'no_helmet' as const,
      event: makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(80) }),
    },
  ];

  it('jumps to the next violation from the start', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId={null} onSelect={onSelect} />,
    );
    await user.click(screen.getByRole('button', { name: 'Next violation' }));
    expect(onSelect).toHaveBeenCalledWith('a');
  });

  it('marks the active marker with aria-current', () => {
    renderInPlayer(
      <Timeline markers={markers} duration={100} selectedEventId="b" onSelect={vi.fn()} />,
    );
    const active = screen.getByRole('button', { name: /No helmet at 1:20/ });
    expect(active).toHaveAttribute('aria-current', 'true');
  });
});
