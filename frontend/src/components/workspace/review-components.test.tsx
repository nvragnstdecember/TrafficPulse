import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { DEFAULT_EVENT_FILTERS } from '@/lib/workspace';
import {
  makeConfirmedEvent,
  makeEvidence,
  makeRenderedEvidence,
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
      filters: DEFAULT_EVENT_FILTERS,
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

describe('EvidenceViewer (H7E, backend artifacts in H14)', () => {
  const event = makeWorkspaceEvent({ trigger_at: mediaSeconds(20) });

  it('displays the backend-rendered frames, never a locally inferred one', () => {
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeRenderedEvidence()} />,
    );
    expect(screen.getByRole('group', { name: 'Evidence frames' })).toBeInTheDocument();
    // The trigger frame is shown by default, sourced from the artifact endpoint.
    const image = screen.getByRole('img', { name: /Evidence frame: Trigger/ });
    expect(image).toHaveAttribute('src', `/api/evidence/${event.id}/artifacts/trigger_frame`);
    // No <video> is involved any more — evidence pixels come from the backend only.
    expect(document.querySelector('video')).toBeNull();
    expect(screen.getByRole('button', { name: 'Copy evidence ID' })).toBeInTheDocument();
  });

  it('offers only the frames the manifest actually has rendered', () => {
    renderWithProviders(
      <EvidenceViewer
        open
        onOpenChange={vi.fn()}
        event={event}
        evidence={makeRenderedEvidence({ after_frame: null })}
      />,
    );
    expect(screen.getByRole('button', { name: 'Before' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Trigger' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'After' })).not.toBeInTheDocument();
  });

  it('switches frames', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeRenderedEvidence()} />,
    );
    await user.click(screen.getByRole('button', { name: 'Before' }));
    expect(screen.getByRole('button', { name: 'Before' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('img', { name: /Evidence frame: Before/ })).toHaveAttribute(
      'src',
      `/api/evidence/${event.id}/artifacts/before_frame`,
    );
  });

  it('zooms in and resets', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeRenderedEvidence()} />,
    );
    expect(screen.getByText('1.0×')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('1.5×')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Reset view' }));
    expect(screen.getByText('1.0×')).toBeInTheDocument();
  });

  it('links the downloadable evidence package', () => {
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeRenderedEvidence()} />,
    );
    const link = screen.getByRole('link', { name: /Download evidence package/ });
    expect(link).toHaveAttribute('href', `/api/evidence/${event.id}/package`);
    expect(link).toHaveAttribute('download');
  });

  it('says so when a pre-H14 manifest has no rendered frames', () => {
    // makeEvidence is the pre-render shape: references without content hashes.
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeEvidence()} />,
    );
    expect(screen.getByText(/No rendered evidence frames exist/)).toBeInTheDocument();
    expect(screen.getByText(/never rendered/)).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /Evidence frame/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
    // The package is still downloadable — metadata is evidence too.
    expect(screen.getByRole('link', { name: /Download evidence package/ })).toBeInTheDocument();
  });

  it('reports a frame that fails to load instead of showing a broken image', () => {
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={makeRenderedEvidence()} />,
    );
    fireEvent.error(screen.getByRole('img', { name: /Evidence frame: Trigger/ }));
    expect(screen.getByText(/could not be loaded/)).toBeInTheDocument();
  });

  it('still shows metadata when the manifest is unavailable', () => {
    renderWithProviders(
      <EvidenceViewer open onOpenChange={vi.fn()} event={event} evidence={undefined} />,
    );
    expect(screen.getByText(/manifest is not available/)).toBeInTheDocument();
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

  it('renders the analyst decision surface as a Review tab', async () => {
    // Notes and decisions moved to the backend in H9, so the detail panel no
    // longer owns any review state — it hosts whatever the workspace injects.
    const user = userEvent.setup();
    renderWithProviders(
      <EventDetail
        event={event}
        detail={makeConfirmedEvent()}
        evidence={undefined}
        isLoading={false}
        onSeek={vi.fn()}
        reviewStatus="approved"
        reviewSlot={<p>decision surface</p>}
      />,
    );

    expect(screen.getByText('Approved')).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Review' }));
    expect(screen.getByText('decision surface')).toBeInTheDocument();
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
      <Timeline markers={markers} selectedEventId={null} onSelect={onSelect} />,
    );
    await user.click(screen.getByRole('button', { name: 'Next violation' }));
    expect(onSelect).toHaveBeenCalledWith('a');
  });

  it('marks the active marker with aria-current', () => {
    renderInPlayer(
      <Timeline markers={markers} selectedEventId="b" onSelect={vi.fn()} />,
    );
    const active = screen.getByRole('button', { name: /No helmet at 1:20/ });
    expect(active).toHaveAttribute('aria-current', 'true');
  });
});
