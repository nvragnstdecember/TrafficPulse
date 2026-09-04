import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { makeComparison, makeExpectation } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { ExpectationPanel } from './expectation-panel';

vi.mock('@/services/demo.service', () => ({
  demoService: {
    getExpectation: vi.fn(),
    declare: vi.fn(),
    withdraw: vi.fn(),
    comparison: vi.fn(),
  },
}));

const { demoService } = await import('@/services/demo.service');

function notDeclared() {
  return new ApiError('no expectation', {
    kind: 'http',
    status: 404,
    type: 'expectation_not_found',
  });
}

beforeEach(() => {
  vi.mocked(demoService.getExpectation).mockRejectedValue(notDeclared());
  vi.mocked(demoService.declare).mockResolvedValue(makeExpectation());
  vi.mocked(demoService.withdraw).mockResolvedValue(undefined);
  vi.mocked(demoService.comparison).mockResolvedValue(makeComparison());
});

function render(overrides: Partial<React.ComponentProps<typeof ExpectationPanel>> = {}) {
  const props: React.ComponentProps<typeof ExpectationPanel> = {
    videoId: 'vid-1',
    jobId: 'job-1',
    runComplete: true,
    ...overrides,
  };
  return renderWithProviders(<ExpectationPanel {...props} />);
}

describe('ExpectationPanel — declaring', () => {
  it('says plainly that a declaration is not a detection', async () => {
    render({ runComplete: false });

    expect(
      await screen.findByText(/never shown to the reasoners/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/never appears in the event list/i)).toBeInTheDocument();
  });

  it('reports an undeclared video as a state, not an error', async () => {
    render({ runComplete: false });

    expect(await screen.findByText('Nothing declared')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('declares the families the analyst selected', async () => {
    const user = userEvent.setup();
    render({ runComplete: false });

    await user.click(await screen.findByRole('button', { name: 'Wrong way' }));
    await user.click(screen.getByRole('button', { name: 'Triple riding' }));
    await user.click(screen.getByRole('button', { name: /^declare$/i }));

    await waitFor(() => expect(demoService.declare).toHaveBeenCalled());
    const [videoId, declaration] = vi.mocked(demoService.declare).mock.calls[0];
    expect(videoId).toBe('vid-1');
    expect(declaration.expected_violations).toEqual(['wrong_way', 'triple_riding']);
  });

  it('records the written test context with the declaration', async () => {
    const user = userEvent.setup();
    render({ runComplete: false });

    await user.type(
      await screen.findByLabelText(/test context/i),
      'Synthetic clip; the zone is artificially designated.',
    );
    await user.click(screen.getByRole('button', { name: /^declare$/i }));

    await waitFor(() => expect(demoService.declare).toHaveBeenCalled());
    const [, declaration] = vi.mocked(demoService.declare).mock.calls[0];
    expect(declaration.notes).toMatch(/artificially designated/);
  });

  it('accepts an empty declaration as a real claim', async () => {
    const user = userEvent.setup();
    render({ runComplete: false });

    await user.type(await screen.findByLabelText(/test context/i), 'Clean traffic.');

    expect(
      screen.getByText(/it claims the clip contains no violation/i),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /^declare$/i }));

    await waitFor(() => expect(demoService.declare).toHaveBeenCalled());
    expect(vi.mocked(demoService.declare).mock.calls[0][1].expected_violations).toEqual([]);
  });

  it('does not offer to save an unchanged declaration', async () => {
    vi.mocked(demoService.getExpectation).mockResolvedValue(makeExpectation());
    render({ runComplete: false });

    const button = await screen.findByRole('button', { name: /update declaration/i });
    expect(button).toBeDisabled();
  });

  it('shows who declared it and when', async () => {
    vi.mocked(demoService.getExpectation).mockResolvedValue(
      makeExpectation({ declared_by: 'ammar' }),
    );
    render({ runComplete: false });

    expect(await screen.findByText(/declared by ammar/i)).toBeInTheDocument();
  });

  it('withdraws a declaration without touching the events', async () => {
    const user = userEvent.setup();
    vi.mocked(demoService.getExpectation).mockResolvedValue(makeExpectation());
    render({ runComplete: false });

    await user.click(await screen.findByRole('button', { name: /withdraw/i }));

    await waitFor(() => expect(demoService.withdraw).toHaveBeenCalledWith('vid-1'));
  });

  it('surfaces a failed declaration rather than swallowing it', async () => {
    const user = userEvent.setup();
    vi.mocked(demoService.declare).mockRejectedValue(
      new ApiError('nope', { kind: 'http', status: 422, type: 'validation_error' }),
    );
    render({ runComplete: false });

    await user.click(await screen.findByRole('button', { name: 'Wrong way' }));
    await user.click(screen.getByRole('button', { name: /^declare$/i }));

    expect(await screen.findByText(/could not save the declaration/i)).toBeInTheDocument();
  });
});

describe('ExpectationPanel — expected vs detected', () => {
  it('withholds the comparison until a run has finished', async () => {
    render({ runComplete: false });

    expect(
      await screen.findByText(/appears once a run has finished/i),
    ).toBeInTheDocument();
    expect(demoService.comparison).not.toHaveBeenCalled();
  });

  it('puts the declared families beside the confirmed events', async () => {
    render();

    const table = await screen.findByRole('table');
    const matched = within(table).getByRole('row', { name: /triple riding/i });
    expect(within(matched).getByText('Declared')).toBeInTheDocument();
    expect(within(matched).getByText('Matched')).toBeInTheDocument();

    const missing = within(table).getByRole('row', { name: /wrong way/i });
    expect(within(missing).getByText('Not detected')).toBeInTheDocument();
  });

  it('summarises with counts and never with an accuracy', async () => {
    render();

    expect(await screen.findByText(/1 of 2 declared families matched/)).toBeInTheDocument();
    expect(
      screen.getByText(/No precision, recall or F1 is shown/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it('explains that a not-detected family is not a defect on its own', async () => {
    render();

    expect(await screen.findByText(/not a defect/i)).toBeInTheDocument();
  });

  it('opens the confirmed event behind a detected count', async () => {
    const user = userEvent.setup();
    const onOpenEvent = vi.fn();
    render({ onOpenEvent });

    await user.click(await screen.findByRole('button', { name: /open #1/i }));

    expect(onOpenEvent).toHaveBeenCalledWith('evt-1');
  });

  it('flags a run that differs from what was declared', async () => {
    vi.mocked(demoService.getExpectation).mockResolvedValue(makeExpectation());
    render();

    expect(await screen.findByText(/differs from declaration/i)).toBeInTheDocument();
  });

  it('reports a run that matched the declaration exactly', async () => {
    vi.mocked(demoService.getExpectation).mockResolvedValue(makeExpectation());
    vi.mocked(demoService.comparison).mockResolvedValue(
      makeComparison({
        missing_count: 0,
        matched_count: 2,
        detected_event_count: 2,
        rows: [
          {
            violation_type: 'triple_riding',
            expected: true,
            detected_count: 1,
            event_ids: ['evt-1'],
            outcome: 'matched',
          },
          {
            violation_type: 'wrong_way',
            expected: true,
            detected_count: 1,
            event_ids: ['evt-2'],
            outcome: 'matched',
          },
        ],
      }),
    );
    render();

    expect(await screen.findByText('As declared')).toBeInTheDocument();
  });

  it('reports an undeclared clip’s detections as unexpected, not as a match', async () => {
    vi.mocked(demoService.comparison).mockResolvedValue(
      makeComparison({
        expectation: null,
        expected_count: 0,
        matched_count: 0,
        missing_count: 0,
        unexpected_count: 1,
        rows: [
          {
            violation_type: 'triple_riding',
            expected: false,
            detected_count: 1,
            event_ids: ['evt-1'],
            outcome: 'unexpected',
          },
        ],
      }),
    );
    render();

    expect(await screen.findByText(/Nothing declared for this clip/)).toBeInTheDocument();
    const row = within(screen.getByRole('table')).getByRole('row', { name: /triple riding/i });
    expect(within(row).getByText('Unexpected')).toBeInTheDocument();
    expect(within(row).getByText('—')).toBeInTheDocument();
  });

  it('scopes the comparison to the run being reviewed', async () => {
    render({ jobId: 'job-42' });

    await waitFor(() =>
      expect(demoService.comparison).toHaveBeenCalledWith('vid-1', 'job-42', expect.anything()),
    );
  });
});
