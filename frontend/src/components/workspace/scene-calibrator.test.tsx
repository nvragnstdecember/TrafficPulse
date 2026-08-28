import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { type SignalPhaseSpec } from '@/api/types';
import { makeSceneSummary } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { SceneCalibrator } from './scene-calibrator';

vi.mock('@/services/scenes.service', () => ({
  scenesService: {
    getForVideo: vi.fn(),
    calibrate: vi.fn(),
    validate: vi.fn(),
  },
}));

const { scenesService } = await import('@/services/scenes.service');

const FRAME = { width: 320, height: 240 };

beforeEach(() => {
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
});

function render(overrides: Partial<React.ComponentProps<typeof SceneCalibrator>> = {}) {
  const onScheduleChange = vi.fn();
  const props: React.ComponentProps<typeof SceneCalibrator> = {
    videoId: 'vid-1',
    frameWidth: FRAME.width,
    frameHeight: FRAME.height,
    posterSrc: 'blob:local',
    schedule: [],
    onScheduleChange,
    ...overrides,
  };
  return { ...renderWithProviders(<SceneCalibrator {...props} />), onScheduleChange, props };
}

/**
 * Click the drawing surface at a scene coordinate.
 *
 * jsdom reports a zero-size bounding box, so the component's ratio maths would map
 * every click to (0,0). Stubbing the rect is what makes the coordinate mapping —
 * the thing actually under test — observable.
 */
function clickAt(surface: Element, x: number, y: number) {
  vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
    left: 0,
    top: 0,
    width: FRAME.width,
    height: FRAME.height,
    right: FRAME.width,
    bottom: FRAME.height,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
  fireEvent.click(surface, { clientX: x, clientY: y });
}

function drawPolygon(surface: Element, points: Array<[number, number]>) {
  for (const [x, y] of points) clickAt(surface, x, y);
}

describe('SceneCalibrator', () => {
  it('reports an uncalibrated video as a state, not an error', async () => {
    render();

    expect(await screen.findByText('Not calibrated')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('cannot save until a lane has been drawn', async () => {
    render();
    const save = await screen.findByRole('button', { name: /save scene/i });
    expect(save).toBeDisabled();

    const surface = screen.getByTestId('calibration-surface');
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);

    await waitFor(() => expect(save).toBeEnabled());
  });

  it('maps a click to the video’s own pixel space and submits it', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');

    drawPolygon(surface, [
      [10, 20],
      [300, 20],
      [300, 200],
    ]);
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    const [, draft] = vi.mocked(scenesService.calibrate).mock.calls[0];
    expect(draft.frame_width).toBe(320);
    expect(draft.zones[0].polygon).toEqual([
      [10, 20],
      [300, 20],
      [300, 200],
    ]);
  });

  it('switches tools and keeps each shape separate', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);

    await user.click(screen.getByRole('button', { name: 'No-stopping zone' }));
    drawPolygon(surface, [
      [10, 10],
      [100, 10],
      [100, 100],
    ]);
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    const [, draft] = vi.mocked(scenesService.calibrate).mock.calls[0];
    expect(draft.zones.map((z) => z.zone_type)).toEqual(['lane', 'no_stopping']);
  });

  it('derives the stop line’s entry direction from the junction', async () => {
    // The analyst draws one line; which way crossing it counts is computed, because
    // an arrow pointed the wrong way makes red-light silently never fire.
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);

    await user.click(screen.getByRole('button', { name: 'Junction area' }));
    drawPolygon(surface, [
      [100, 150],
      [220, 150],
      [220, 235],
    ]);
    await user.click(screen.getByRole('button', { name: 'Stop line' }));
    clickAt(surface, 100, 120);
    clickAt(surface, 220, 120);
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    const [, draft] = vi.mocked(scenesService.calibrate).mock.calls[0];
    expect(draft.stop_lines?.[0].crossing_dy).toBeGreaterThan(0); // down, into the junction
  });

  it('clears only the active tool’s shape', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /save scene/i })).toBeEnabled(),
    );

    await user.click(screen.getByRole('button', { name: /clear lane/i }));

    expect(screen.getByRole('button', { name: /save scene/i })).toBeDisabled();
  });

  it('shows the contract’s own messages when a draft is refused', async () => {
    const user = userEvent.setup();
    vi.mocked(scenesService.validate).mockResolvedValue({
      valid: false,
      errors: ['zones.0.polygon: image points out of frame bounds'],
      supported_violations: [],
      scene_hash: null,
    });
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);

    await user.click(screen.getByRole('button', { name: /check/i }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('out of frame bounds');
  });

  it('shows what a saved scene unlocks', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['wrong_way', 'red_light_jumping'] }),
    );
    render();

    expect(await screen.findByText(/Calibrated/)).toBeInTheDocument();
    const unlocked = screen.getByLabelText('Violations unlocked');
    expect(within(unlocked).getByText('Wrong way')).toBeInTheDocument();
    expect(within(unlocked).getByText('Red-light jumping')).toBeInTheDocument();
  });

  it('offers a re-run once a scene is saved', async () => {
    const user = userEvent.setup();
    const onReprocess = vi.fn();
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['wrong_way'] }),
    );
    render({ onReprocess });

    await user.click(await screen.findByRole('button', { name: /re-run analysis/i }));

    expect(onReprocess).toHaveBeenCalledWith(['wrong_way']);
  });

  it('renders without a backdrop when the video is not playable here', async () => {
    render({ posterSrc: null });

    expect(await screen.findByTestId('calibration-no-backdrop')).toBeInTheDocument();
    expect(screen.getByTestId('calibration-surface')).toBeInTheDocument();
  });
});

describe('SignalScheduleEditor (via the calibrator)', () => {
  function renderWithJunction(schedule: SignalPhaseSpec[] = []) {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['red_light_jumping'] }),
    );
    return render({ schedule });
  }

  it('is hidden for a scene with no junction geometry', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['wrong_way'] }),
    );
    render();

    await screen.findByText(/Calibrated/);
    expect(screen.queryByLabelText('Signal schedule')).not.toBeInTheDocument();
  });

  it('appears once red-light is unlocked, and says the state is declared', async () => {
    renderWithJunction();

    const section = await screen.findByLabelText('Signal schedule');
    expect(within(section).getByText(/declared, not detected/i)).toBeInTheDocument();
    expect(within(section).getByText(/needs at least one/i)).toBeInTheDocument();
  });

  it('adds a phase', async () => {
    const user = userEvent.setup();
    const { onScheduleChange } = renderWithJunction();

    await user.click(await screen.findByRole('button', { name: /add phase/i }));

    expect(onScheduleChange).toHaveBeenCalledWith([{ at_seconds: 0, state: 'red' }]);
  });

  it('edits a phase’s time and state', async () => {
    const user = userEvent.setup();
    const { onScheduleChange } = renderWithJunction([{ at_seconds: 0, state: 'red' }]);

    await user.selectOptions(await screen.findByLabelText('Phase 1 state'), 'green');

    expect(onScheduleChange).toHaveBeenCalledWith([{ at_seconds: 0, state: 'green' }]);
  });

  it('removes a phase', async () => {
    const user = userEvent.setup();
    const { onScheduleChange } = renderWithJunction([
      { at_seconds: 0, state: 'red' },
      { at_seconds: 9, state: 'green' },
    ]);

    await user.click(await screen.findByRole('button', { name: 'Remove phase 1' }));

    expect(onScheduleChange).toHaveBeenCalledWith([{ at_seconds: 9, state: 'green' }]);
  });

  it('warns about an out-of-order schedule', async () => {
    renderWithJunction([
      { at_seconds: 10, state: 'red' },
      { at_seconds: 2, state: 'green' },
    ]);

    expect(await screen.findByText(/non-decreasing time order/i)).toBeInTheDocument();
  });
});

/**
 * A scene nobody drew has to announce itself, and has to be honest about which
 * outcome it was. The regression this guards: the banner used to claim, for every
 * derived scene, that "the legal direction was estimated from the traffic in this
 * clip" — which is false whenever derivation abstained, and abstention is the
 * expected outcome on a two-way road.
 */
describe('SceneCalibrator — derived-scene provenance', () => {
  it('says a legal direction was estimated when one actually was', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ derived: true, has_legal_direction: true }),
    );

    render();

    const note = await screen.findByRole('note');
    expect(note).toHaveTextContent(/derived automatically/i);
    expect(note).toHaveTextContent(/estimated/i);
    expect(note).toHaveTextContent(/no no-stopping zone, stop line or signal timing/i);
  });

  it('says no direction could be established when derivation abstained', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({
        derived: true,
        has_legal_direction: false,
        supported_violations: ['triple_riding'],
      }),
    );

    render();

    const note = await screen.findByRole('note');
    expect(note).toHaveTextContent(/no legal direction could be established/i);
    // The false claim the old copy made unconditionally.
    expect(note).not.toHaveTextContent(/direction estimated from/i);
    expect(note).toHaveTextContent(/wrong-way detection is therefore unavailable/i);
  });

  it('shows no provenance note at all for a scene an analyst drew', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ derived: false, has_legal_direction: true }),
    );

    render();

    // The saved scene has loaded (its unlocked violations are on screen)…
    expect(await screen.findByLabelText('Violations unlocked')).toBeInTheDocument();
    // …and nothing casts doubt on geometry a person actually drew.
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
  });
});
