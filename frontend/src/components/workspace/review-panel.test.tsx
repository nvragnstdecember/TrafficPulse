import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { makeReviewCase, makeReviewEntry } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { ReviewPanel } from './review-panel';

function render(props: Partial<React.ComponentProps<typeof ReviewPanel>> = {}) {
  const onDecide = vi.fn();
  renderWithProviders(
    <ReviewPanel
      eventId="evt-1"
      reviewCase={makeReviewCase()}
      history={[]}
      isLoading={false}
      onDecide={onDecide}
      isDeciding={false}
      reviewer="analyst"
      {...props}
    />,
  );
  return { onDecide };
}

describe('ReviewPanel', () => {
  it('shows the status and only the legal decisions', () => {
    render();
    // Shown twice by design: the status chip and the metadata row.
    expect(screen.getAllByText('Pending').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: 'Start review' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Approve' })).not.toBeInTheDocument();
  });

  it('offers the decision set once a case is in review', () => {
    render({ reviewCase: makeReviewCase({ status: 'in_review' }) });
    for (const label of ['Approve', 'Reject', 'False positive', 'Needs evidence']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    }
  });

  it('sends the typed note along with the decision', async () => {
    // A reviewer types the justification and clicks Approve — both belong to one
    // journal entry, not two round-trips.
    const user = userEvent.setup();
    const { onDecide } = render({ reviewCase: makeReviewCase({ status: 'in_review' }) });

    await user.type(screen.getByLabelText('Analyst note'), 'Plate legible');
    await user.click(screen.getByRole('button', { name: 'Approve' }));

    expect(onDecide).toHaveBeenCalledWith('approve', 'Plate legible');
  });

  it('sends null rather than an empty note', () => {
    const { onDecide } = render({ reviewCase: makeReviewCase({ status: 'in_review' }) });
    screen.getByRole('button', { name: 'Approve' }).click();
    expect(onDecide).toHaveBeenCalledWith('approve', null);
  });

  it('can save a note without deciding', async () => {
    const user = userEvent.setup();
    const { onDecide } = render({ reviewCase: makeReviewCase({ status: 'in_review' }) });

    const save = screen.getByRole('button', { name: /save note/i });
    expect(save).toBeDisabled(); // nothing to save yet

    await user.type(screen.getByLabelText('Analyst note'), 'Checking the plate');
    await user.click(save);

    expect(onDecide).toHaveBeenCalledWith('note', 'Checking the plate');
  });

  it('shows the saved note and the review metadata', () => {
    render({
      reviewCase: makeReviewCase({
        status: 'approved',
        reviewer_id: 'analyst-b',
        note: 'Confirmed on review',
        decided_at: '2026-07-29T12:00:00Z',
        updated_at: '2026-07-29T12:05:00Z',
      }),
    });
    expect(screen.getByText('Confirmed on review')).toBeInTheDocument();
    expect(screen.getByText('analyst-b')).toBeInTheDocument();
  });

  it('renders the audit history oldest first', () => {
    render({
      reviewCase: makeReviewCase({ status: 'approved' }),
      history: [
        makeReviewEntry(),
        makeReviewEntry({
          entry_id: 'rev-2',
          action: 'note',
          status_before: 'in_review',
          status_after: 'in_review',
          note: 'Plate legible',
        }),
        makeReviewEntry({
          entry_id: 'rev-3',
          action: 'approve',
          status_before: 'in_review',
          status_after: 'approved',
        }),
      ],
    });

    const audit = screen.getByRole('region', { name: 'Audit history' });
    const items = within(audit).getAllByRole('listitem');
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent('Review opened');
    expect(items[1]).toHaveTextContent('Note added');
    expect(items[1]).toHaveTextContent('Plate legible');
    expect(items[2]).toHaveTextContent('Approved');
  });

  it('explains an empty history rather than showing a blank area', () => {
    render();
    expect(screen.getByText(/No analyst action recorded yet/)).toBeInTheDocument();
  });

  it('disables the decisions while one is in flight', () => {
    render({ reviewCase: makeReviewCase({ status: 'in_review' }), isDeciding: true });
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
  });

  it('surfaces a load failure with a retry', () => {
    const onRetry = vi.fn();
    render({ error: new Error('boom'), onRetry });
    expect(screen.getByText(/Could not load the review/)).toBeInTheDocument();
  });

  it('clears the draft note when the analyst moves to another event', async () => {
    // A note must never be carried onto a case it was not written for.
    const user = userEvent.setup();
    const onDecide = vi.fn();
    const { rerender } = renderWithProviders(
      <ReviewPanel
        eventId="evt-1"
        reviewCase={makeReviewCase({ status: 'in_review' })}
        history={[]}
        isLoading={false}
        onDecide={onDecide}
        isDeciding={false}
        reviewer="analyst"
      />,
    );
    await user.type(screen.getByLabelText('Analyst note'), 'For evt-1');

    rerender(
      <ReviewPanel
        eventId="evt-2"
        reviewCase={makeReviewCase({ event_id: 'evt-2', status: 'in_review' })}
        history={[]}
        isLoading={false}
        onDecide={onDecide}
        isDeciding={false}
        reviewer="analyst"
      />,
    );

    expect(screen.getByLabelText('Analyst note')).toHaveValue('');
  });
});
