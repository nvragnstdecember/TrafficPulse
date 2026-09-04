import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { type SignalPhaseSpec } from '@/api/types';
import { makeSceneSummary, makeStoredScene } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

import { SceneCalibrator } from './scene-calibrator';

vi.mock('@/services/scenes.service', () => ({
  scenesService: {
    get: vi.fn(),
    getForVideo: vi.fn(),
    calibrate: vi.fn(),
    validate: vi.fn(),
  },
}));

const { scenesService } = await import('@/services/scenes.service');

const FRAME = { width: 320, height: 240 };

beforeEach(() => {
  vi.mocked(scenesService.get).mockRejectedValue(
    new ApiError('no scene', { kind: 'http', status: 404, type: 'scene_not_found' }),
  );
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

describe('SceneCalibrator — declared context (controlled demo)', () => {
  function drawLane(surface: Element) {
    drawPolygon(surface, [
      [0, 0],
      [320, 0],
      [320, 240],
    ]);
  }

  it('says the context is declared rather than inferred', async () => {
    // The single most important sentence on this panel: a viewer who assumes the
    // system worked out the legal direction has misunderstood the whole system.
    render();

    const notice = await screen.findByTestId('calibration-declared-notice');
    expect(notice).toHaveTextContent(/Everything here is declared/i);
    expect(notice).toHaveTextContent(/not things the camera can establish from pixels/i);
    expect(notice).toHaveTextContent(/does not infer it/i);
  });

  it('sends the dwell threshold the analyst typed', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawLane(surface);

    await user.click(screen.getByRole('button', { name: 'No-stopping zone' }));
    drawPolygon(surface, [
      [10, 10],
      [100, 10],
      [100, 100],
    ]);
    await user.clear(screen.getByLabelText(/stopping dwell/i));
    await user.type(screen.getByLabelText(/stopping dwell/i), '8');
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    const [, draft] = vi.mocked(scenesService.calibrate).mock.calls[0];
    expect(draft.tuning).toEqual({ stationary_duration_seconds: 8 });
  });

  it('sends no tuning at all when the analyst typed nothing', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawLane(surface);
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    expect(vi.mocked(scenesService.calibrate).mock.calls[0][1].tuning).toBeUndefined();
  });

  it('refuses to save a threshold the backend would reject', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawLane(surface);
    await user.click(screen.getByRole('button', { name: 'No-stopping zone' }));
    drawPolygon(surface, [
      [10, 10],
      [100, 10],
      [100, 100],
    ]);

    await user.clear(screen.getByLabelText(/stopping dwell/i));
    await user.type(screen.getByLabelText(/stopping dwell/i), '0');

    expect(await screen.findByText(/must be greater than 0/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save scene/i })).toBeDisabled();
  });

  it('offers a dwell threshold only once a no-stopping zone exists', async () => {
    render();

    expect(
      await screen.findByText(/draw a no-stopping zone to set a dwell threshold/i),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText(/stopping dwell/i)).not.toBeInTheDocument();
  });

  it('stores the analyst’s scene notes inside the scene', async () => {
    const user = userEvent.setup();
    render();
    const surface = await screen.findByTestId('calibration-surface');
    drawLane(surface);

    await user.type(
      screen.getByLabelText(/scene notes/i),
      'Controlled demo; the zone is designated for demonstration.',
    );
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    expect(vi.mocked(scenesService.calibrate).mock.calls[0][1].description).toMatch(
      /designated for demonstration/,
    );
  });
});

describe('SceneCalibrator — reloading a saved calibration', () => {
  it('offers nothing to load for an uncalibrated video', async () => {
    render();

    await screen.findByText('Not calibrated');
    expect(
      screen.queryByRole('button', { name: /load saved calibration/i }),
    ).not.toBeInTheDocument();
  });

  it('redraws the bound revision onto the surface', async () => {
    // Reproducibility: before this, a saved calibration was write-only — refresh the
    // page and nobody could see what had been drawn, or correct one polygon.
    const user = userEvent.setup();
    vi.mocked(scenesService.getForVideo).mockResolvedValue(makeSceneSummary());
    vi.mocked(scenesService.get).mockResolvedValue(makeStoredScene());
    render();

    await user.click(await screen.findByRole('button', { name: /load saved calibration/i }));
    await user.click(screen.getByRole('button', { name: /save scene/i }));

    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    const [, draft] = vi.mocked(scenesService.calibrate).mock.calls[0];
    expect(draft.zones.map((z) => z.zone_id)).toEqual(['zone-lane', 'zone-no-stopping']);
    expect(draft.tuning).toEqual({ stationary_duration_seconds: 7 });
    expect(draft.description).toMatch(/Controlled demonstration/);
  });

  it('loads only when asked, so an in-progress drawing is never clobbered', async () => {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(makeSceneSummary());
    vi.mocked(scenesService.get).mockResolvedValue(makeStoredScene());
    render();

    const surface = await screen.findByTestId('calibration-surface');
    drawPolygon(surface, [
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    // The revision has resolved by now, but the drawing is untouched until asked.
    await screen.findByRole('button', { name: /load saved calibration/i });

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /save scene/i }));
    await waitFor(() => expect(scenesService.calibrate).toHaveBeenCalled());
    expect(vi.mocked(scenesService.calibrate).mock.calls[0][1].zones[0].polygon).toEqual([
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
  });
});

describe('SceneCalibrator — representative frame', () => {
  it('offers a frame picker when the clip is playable and its length is known', async () => {
    render({ durationSeconds: 12 });

    expect(await screen.findByLabelText(/representative frame/i)).toBeInTheDocument();
  });

  it('offers none when there is nothing to scrub', async () => {
    render({ posterSrc: null, durationSeconds: 12 });

    await screen.findByTestId('calibration-no-backdrop');
    expect(screen.queryByLabelText(/representative frame/i)).not.toBeInTheDocument();
  });

  it('says the geometry applies to the whole clip regardless of the frame drawn on', async () => {
    render({ durationSeconds: 12 });

    expect(
      await screen.findByText(/applies to the whole clip regardless of which frame/i),
    ).toBeInTheDocument();
  });
});

describe('SignalScheduleEditor — declared timeline', () => {
  function renderWithJunction(schedule: SignalPhaseSpec[] = [], durationSeconds = 20) {
    vi.mocked(scenesService.getForVideo).mockResolvedValue(
      makeSceneSummary({ supported_violations: ['red_light_jumping'] }),
    );
    return render({ schedule, durationSeconds });
  }

  it('draws the declared phases over the clip', async () => {
    renderWithJunction([
      { at_seconds: 0, state: 'red' },
      { at_seconds: 8, state: 'green' },
    ]);

    const timeline = await screen.findByTestId('signal-timeline');
    expect(within(timeline).getByRole('img')).toHaveAccessibleName(
      /Red from 0.0s, Green from 8.0s/,
    );
  });

  it('warns when the clip opens with no declared state', async () => {
    // Before the first phase every instant resolves to `unknown`, and red-light can
    // never confirm there — a gap that is invisible in a list of numbers.
    renderWithJunction([{ at_seconds: 5, state: 'red' }]);

    expect(
      await screen.findByText(/The clip opens with no declared state/i),
    ).toBeInTheDocument();
  });

  it('does not warn when the schedule starts at zero', async () => {
    renderWithJunction([{ at_seconds: 0, state: 'red' }]);

    await screen.findByTestId('signal-timeline');
    expect(screen.queryByText(/opens with no declared state/i)).not.toBeInTheDocument();
  });

  it('draws no timeline when the clip length is unknown', async () => {
    renderWithJunction([{ at_seconds: 0, state: 'red' }], 0);

    await screen.findByLabelText('Signal schedule');
    expect(screen.queryByTestId('signal-timeline')).not.toBeInTheDocument();
  });
});
