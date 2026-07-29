import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { stageStates } from '@/lib/job';
import { buildEventNarrative } from '@/lib/workspace';
import { makeJob, makeWorkspaceEvent, mediaSeconds } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { EventNarrative } from './event-narrative';
import { ReviewSummary } from './review-summary';
import { StatGrid } from './stat-grid';
import { WorkflowNav } from './workflow-nav';

describe('StatGrid', () => {
  it('renders an unmeasured value as a dash, never as zero', () => {
    renderWithProviders(
      <StatGrid
        stats={[
          { key: 'a', label: 'Detections', value: null },
          { key: 'b', label: 'Violations', value: '3' },
        ]}
      />,
    );
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });
});

describe('EventNarrative', () => {
  const event = makeWorkspaceEvent({
    start_at: mediaSeconds(20),
    trigger_at: mediaSeconds(24),
  });

  it('lists the stages with their timestamps', () => {
    renderWithProviders(
      <EventNarrative steps={buildEventNarrative(event, { hasEvidence: true })} />,
    );
    expect(screen.getByText('Violation confirmed')).toBeInTheDocument();
    expect(screen.getByText('Evidence package finalized')).toBeInTheDocument();
    // Confirmation and evidence share the trigger instant, so the clock repeats.
    expect(screen.getAllByText('0:24').length).toBeGreaterThan(0);
  });

  it('seeks the player when a step is picked', async () => {
    const onSeek = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <EventNarrative steps={buildEventNarrative(event)} onSeek={onSeek} />,
    );

    await user.click(screen.getByText('Violation confirmed'));

    expect(onSeek).toHaveBeenCalledWith(24);
  });

  it('renders nothing for an empty story', () => {
    const { container } = renderWithProviders(<EventNarrative steps={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe('workflow navigation', () => {
  it('derives stages from the processing phase, with no state of its own', () => {
    expect(stageStates('idle', false).upload).toBe('current');
    expect(stageStates('running', false).processing).toBe('current');
    expect(stageStates('completed', false).review).toBe('current');
    expect(stageStates('completed', true).evidence).toBe('current');
    expect(stageStates('completed', true).review).toBe('done');
  });

  it('leaves unreached stages inert rather than hidden', async () => {
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <WorkflowNav phase="idle" hasSelection={false} onNavigate={onNavigate} />,
    );

    // Every stage is visible, so the shape of the workflow is clear from the start.
    expect(screen.getByText('Evidence')).toBeInTheDocument();
    await user.click(screen.getByText('Evidence'));
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it('navigates to a reached stage', async () => {
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <WorkflowNav phase="completed" hasSelection={false} onNavigate={onNavigate} />,
    );

    await user.click(screen.getByText('Upload'));

    expect(onNavigate).toHaveBeenCalledWith('upload');
  });
});

describe('ReviewSummary', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', violation_type: 'no_helmet' }),
    makeWorkspaceEvent({ event_id: 'b', violation_type: 'no_helmet' }),
  ];

  it('shows the completion headline once the run has finished', () => {
    renderWithProviders(
      <ReviewSummary
        events={events}
        metrics={null}
        job={makeJob({ status: 'succeeded' })}
        elapsedSeconds={12}
        models={[{ name: 'rtdetr', version: '1.0', weights_hash: null }]}
        complete
      />,
    );
    expect(screen.getByText('Processing complete')).toBeInTheDocument();
    expect(screen.getByText('No helmet 2')).toBeInTheDocument();
    expect(screen.getByText('rtdetr@1.0')).toBeInTheDocument();
  });

  it('shows the in-flight headline while the run is still going', () => {
    renderWithProviders(
      <ReviewSummary
        events={[]}
        metrics={null}
        job={makeJob()}
        elapsedSeconds={null}
        models={[]}
        complete={false}
      />,
    );
    expect(screen.getByText('Run statistics')).toBeInTheDocument();
  });
});
