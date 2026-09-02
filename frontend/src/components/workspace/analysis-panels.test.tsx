import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { type HelmetAnalysis, type RiderAnalysis, type SystemPosture } from '@/api/types';
import { DEFAULT_EVENT_FILTERS, DEFAULT_WORKSPACE_SORT } from '@/lib/workspace';
import { renderWithProviders } from '@/test/utils';

import { EventList } from './event-list';
import { HelmetAnalysisPanel } from './helmet-analysis-panel';
import { SystemPostureStrip } from './system-posture';

function rider(overrides: Partial<RiderAnalysis> = {}): RiderAnalysis {
  return {
    rider_track_id: 'rider-1',
    motorcycle_track_id: 'bike-1',
    rider_count: 1,
    multi_rider: false,
    samples: 12,
    first_frame: 0,
    last_frame: 11,
    helmet_state: 'helmet',
    confidence: 0.93,
    agreement: 1,
    settled: true,
    raw_label_flips: 0,
    stabilized_label_flips: 0,
    median_head_height_px: 44,
    enforcement: 'eligible',
    ...overrides,
  };
}

function analysis(overrides: Partial<HelmetAnalysis> = {}): HelmetAnalysis {
  return {
    job_id: 'job-1',
    enforcement: 'disabled',
    frames_observed: 40,
    riders_observed: 1,
    motorcycles_associated: 1,
    multi_rider_riders: 0,
    multi_rider_motorcycles: 0,
    eligible_riders: 1,
    unresolved_riders: 0,
    abstained_riders: 0,
    unstable_riders: 0,
    gate_abstentions: 2,
    label_counts: [{ label: 'helmet', riders: 1 }],
    enforcement_counts: [{ label: 'eligible', riders: 1 }],
    riders: [rider()],
    ...overrides,
  };
}

const POSTURE: SystemPosture = {
  components: [
    {
      component_id: 'detection',
      label: 'Detection',
      state: 'limited',
      detail: 'RT-DETR is running; recall is not complete.',
    },
    {
      component_id: 'helmet_enforcement',
      label: 'Helmet violation enforcement',
      state: 'disabled',
      detail: 'This deployment runs helmet classification as analysis only.',
    },
  ],
  helmet_backend: 'ResNetHelmetConfig',
  helmet_backend_labels: ['helmet', 'no_helmet', 'uncertain'],
  turban_capable: false,
  helmet_enforcement: 'disabled',
};

describe('SystemPostureStrip', () => {
  it('renders nothing before the posture is known', () => {
    const { container } = renderWithProviders(<SystemPostureStrip posture={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('states the enforcement reason in full, not just a status word', () => {
    renderWithProviders(<SystemPostureStrip posture={POSTURE} />);
    expect(
      screen.getByText(/runs helmet classification as analysis only/i),
    ).toBeInTheDocument();
  });

  it('says why no exemption can fire when the backend cannot emit turban', () => {
    renderWithProviders(<SystemPostureStrip posture={POSTURE} />);
    expect(screen.getByText(/no turban label/i)).toBeInTheDocument();
  });
});

describe('HelmetAnalysisPanel', () => {
  it('renders nothing when the run declared no analysis', () => {
    // A deployment without an analysis is working as configured, so there is no
    // empty state and no error to show.
    const { container } = renderWithProviders(<HelmetAnalysisPanel analysis={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('states up front that no violation is decided', () => {
    renderWithProviders(<HelmetAnalysisPanel analysis={analysis()} />);
    expect(screen.getByText(/No violation is decided here/i)).toBeInTheDocument();
  });

  it('marks a multi-rider motorcycle unresolved and picks no driver', () => {
    renderWithProviders(
      <HelmetAnalysisPanel
        analysis={analysis({
          riders_observed: 2,
          multi_rider_riders: 2,
          multi_rider_motorcycles: 1,
          eligible_riders: 0,
          unresolved_riders: 2,
          riders: [
            rider({
              rider_track_id: 'rider-1',
              multi_rider: true,
              rider_count: 2,
              helmet_state: 'no_helmet',
              enforcement: 'multi_rider_unresolved',
            }),
            rider({
              rider_track_id: 'rider-2',
              multi_rider: true,
              rider_count: 2,
              enforcement: 'multi_rider_unresolved',
            }),
          ],
        })}
      />,
    );

    expect(screen.getAllByText(/Multi-rider — driver unresolved/i).length).toBe(2);
    expect(screen.getByText(/does not attempt\s+to tell driver from pillion/i)).toBeInTheDocument();
    // Neither rider is singled out as the driver anywhere on the panel.
    expect(screen.queryByText(/\bdriver:/i)).not.toBeInTheDocument();
  });

  it('shows an em dash, never 0%, for a crop the classifier never scored', () => {
    renderWithProviders(
      <HelmetAnalysisPanel
        analysis={analysis({
          riders: [
            rider({
              helmet_state: 'uncertain',
              confidence: null,
              enforcement: 'classification_abstained',
            }),
          ],
        })}
      />,
    );
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });

  it('discloses the observed per-frame instability of this run', () => {
    renderWithProviders(
      <HelmetAnalysisPanel
        analysis={analysis({
          riders: [rider({ raw_label_flips: 4 }), rider({ rider_track_id: 'rider-2' })],
        })}
      />,
    );
    expect(screen.getByText(/1 of 2/)).toBeInTheDocument();
  });

  it('explains an empty clip instead of showing a bare zero', () => {
    renderWithProviders(
      <HelmetAnalysisPanel
        analysis={analysis({
          riders: [],
          riders_observed: 0,
          motorcycles_associated: 0,
          eligible_riders: 0,
          label_counts: [],
          enforcement_counts: [],
        })}
      />,
    );
    expect(screen.getByText(/no motorcycles, that is the correct outcome/i)).toBeInTheDocument();
  });
});


describe('EventList empty state', () => {
  function renderEmpty(enforcementNote: string | null) {
    return renderWithProviders(
      <EventList
        events={[]}
        totalCount={0}
        selectedEventId={null}
        onSelect={() => {}}
        filters={DEFAULT_EVENT_FILTERS}
        onFiltersChange={() => {}}
        sort={DEFAULT_WORKSPACE_SORT}
        onSortChange={() => {}}
        availableViolations={[]}
        isLoading={false}
        isError={false}
        onRetry={() => {}}
        selectionMode={false}
        onSelectionModeChange={() => {}}
        checkedIds={new Set()}
        onToggleChecked={() => {}}
        onCheckAll={() => {}}
        onClearChecked={() => {}}
        onExport={() => {}}
        isProcessing={false}
        enforcementNote={enforcementNote}
      />,
    );
  }

  it('does not claim a finding when a violation family was never allowed to run', () => {
    // The demo's normal state: helmet enforcement off, so an empty list means
    // "not checked", not "checked and clean". Saying "detected" here would convert
    // a deliberate abstention into a finding of compliance.
    renderEmpty('Helmet violation enforcement is off in this deployment.');

    expect(screen.getByText('No violations confirmed')).toBeInTheDocument();
    expect(screen.queryByText('No violations detected')).not.toBeInTheDocument();
    expect(screen.getByText(/enforcement is off/i)).toBeInTheDocument();
  });

  it('keeps the original wording where it is already honest', () => {
    renderEmpty(null);

    expect(screen.getByText('No violations detected')).toBeInTheDocument();
  });
});
