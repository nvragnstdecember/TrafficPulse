import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { buildTimelineMarkers, timelineDuration, toWorkspaceEvent } from '@/lib/workspace';
import { makeEventSummary, mediaSeconds } from '@/test/fixtures';

import { PlayerProvider, usePlayer } from './player-context';
import { Timeline } from './timeline';
import { VideoPlayer } from './video-player';

/**
 * Player + timeline as the workspace actually composes them (regression cover).
 *
 * `use-video-controller.test.tsx` covers the controller in isolation and
 * `workspace.test.tsx` covers the timeline's markers, but nothing covered the
 * thing an analyst watches: whether the **playhead and the seek bar advance** when
 * the media clock does. These tests mount the real `VideoPlayer` and `Timeline`
 * inside one `PlayerProvider` and derive `duration` from the shared controller —
 * exactly what `WorkspaceView` does — then drive the real `<video>` element.
 */

/** The playhead: the full-height bar the timeline positions by percentage. */
function playheadPercent(): number {
  const el = document.querySelector<HTMLElement>('.pointer-events-none.absolute.h-full');
  if (!el) throw new Error('playhead not found');
  return Number.parseFloat(el.style.left);
}

function seekBar(): HTMLInputElement {
  return screen.getByLabelText('Seek') as HTMLInputElement;
}

function videoEl(): HTMLVideoElement {
  return screen.getByLabelText('Video preview') as HTMLVideoElement;
}

/** Drive the media clock the way a real element does: set the property, emit. */
function advanceTo(seconds: number): void {
  const el = videoEl();
  Object.defineProperty(el, 'currentTime', { value: seconds, configurable: true });
  act(() => void el.dispatchEvent(new Event('timeupdate')));
}

function loadMetadata(duration: number): void {
  const el = videoEl();
  Object.defineProperty(el, 'duration', { value: duration, configurable: true });
  act(() => void el.dispatchEvent(new Event('loadedmetadata')));
}

/** Mirrors WorkspaceView: duration comes from the shared controller state. */
function Workspace({ eventSeconds }: { eventSeconds: number[] }) {
  const { state } = usePlayer();
  const events = eventSeconds.map((seconds, index) =>
    toWorkspaceEvent(
      makeEventSummary({ event_id: `evt-${index}`, trigger_at: mediaSeconds(seconds) }),
    ),
  );
  const duration = timelineDuration(state.duration, events);
  return (
    <>
      <VideoPlayer src="blob:clip" />
      <Timeline
        markers={buildTimelineMarkers(events, duration)}
        selectedEventId={null}
        onSelect={() => {}}
      />
    </>
  );
}

function renderWorkspace(eventSeconds: number[] = [10, 30]) {
  return render(
    <PlayerProvider fps={25}>
      <Workspace eventSeconds={eventSeconds} />
    </PlayerProvider>,
  );
}

describe('timeline playback progress', () => {
  it('starts at the origin before metadata arrives', () => {
    renderWorkspace();
    expect(playheadPercent()).toBe(0);
    expect(seekBar()).toBeDisabled();
  });

  it('populates duration from loadedmetadata', () => {
    renderWorkspace();
    loadMetadata(60);
    expect(seekBar()).toBeEnabled();
    expect(seekBar().max).toBe('60');
    expect(screen.getByText('0:00 / 1:00')).toBeInTheDocument();
  });

  it('advances the playhead as the media clock advances', () => {
    renderWorkspace();
    loadMetadata(60);

    advanceTo(15);
    expect(playheadPercent()).toBeCloseTo(25, 1);

    advanceTo(30);
    expect(playheadPercent()).toBeCloseTo(50, 1);

    advanceTo(60);
    expect(playheadPercent()).toBeCloseTo(100, 1);
  });

  it('advances the seek bar with the media clock', () => {
    renderWorkspace();
    loadMetadata(60);
    advanceTo(24);
    expect(seekBar().value).toBe('24');
  });

  it('shows the running clock in the timeline and the controls', () => {
    renderWorkspace();
    loadMetadata(120);
    advanceTo(83);
    expect(screen.getByText('1:23 / 2:00')).toBeInTheDocument();
  });

  it('never drives the playhead past the end', () => {
    renderWorkspace();
    loadMetadata(60);
    advanceTo(120);
    expect(playheadPercent()).toBeLessThanOrEqual(100);
  });

  it('keeps markers positioned against the real duration', () => {
    renderWorkspace([10, 30]);
    loadMetadata(60);
    // 10s and 30s of a 60s clip sit at 1/6 and 1/2 of the track.
    const markers = screen.getAllByRole('button', { name: /at \d/ });
    expect(markers[0].style.left).toBe(`${(10 / 60) * 100}%`);
    expect(markers[1].style.left).toBe('50%');
  });

  it('seeks to the media time the click position names', async () => {
    const user = userEvent.setup();
    renderWorkspace([10, 30]);
    loadMetadata(60);

    // jsdom reports a zero-width rect, so drive the handler with a known geometry.
    const track = document.querySelector<HTMLElement>('[role="presentation"]')!;
    track.getBoundingClientRect = () => ({ left: 0, width: 200 }) as DOMRect;
    await user.pointer({ target: track, coords: { clientX: 100, clientY: 0 }, keys: '[MouseLeft]' });

    // Half-way across a 60s clip is 0:30 — not half-way to the last event.
    expect(videoEl().currentTime).toBe(30);
  });
});

describe('timeline without a media duration (regression)', () => {
  // The failure mode this guards: the layout span falls back to the last event's
  // time, and using it for playback made the timeline report a confident, wrong
  // position. Markers must still lay out; progress must not be invented.

  it('does not scale the playhead against the marker fallback', () => {
    renderWorkspace([10, 30]); // fallback span would be 30s
    // No loadedmetadata: the player has no duration.
    advanceTo(15);
    // 15s of an unknown-length video is not "half way".
    expect(playheadPercent()).toBe(0);
  });

  it('still lays markers out so the events remain visible', () => {
    renderWorkspace([10, 30]);
    const markers = screen.getAllByRole('button', { name: /at \d/ });
    expect(markers).toHaveLength(2);
    expect(markers[1].style.left).toBe('100%');
  });

  it('refuses to seek rather than seeking to a wrong time', async () => {
    const user = userEvent.setup();
    renderWorkspace([10, 30]);

    const track = document.querySelector<HTMLElement>('[role="presentation"]')!;
    track.getBoundingClientRect = () => ({ left: 0, width: 200 }) as DOMRect;
    await user.pointer({ target: track, coords: { clientX: 100, clientY: 0 }, keys: '[MouseLeft]' });

    expect(videoEl().currentTime).toBe(0);
    expect(seekBar()).toBeDisabled();
  });

  it('starts reporting position as soon as metadata arrives', () => {
    renderWorkspace([10, 30]);
    advanceTo(15);
    expect(playheadPercent()).toBe(0);

    loadMetadata(60);
    advanceTo(15);
    expect(playheadPercent()).toBeCloseTo(25, 1);
  });
});

describe('timeline edge cases', () => {
  it('handles a video with no events', () => {
    renderWorkspace([]);
    loadMetadata(60);
    advanceTo(30);
    expect(playheadPercent()).toBeCloseTo(50, 1);
    expect(screen.queryAllByRole('button', { name: /at \d/ })).toHaveLength(0);
  });

  it('handles a very short video', () => {
    renderWorkspace([0.5]);
    loadMetadata(1);
    advanceTo(0.5);
    expect(playheadPercent()).toBeCloseTo(50, 1);
  });

  it('handles a long video', () => {
    renderWorkspace([3600]);
    loadMetadata(7200);
    advanceTo(1800);
    expect(playheadPercent()).toBeCloseTo(25, 1);
    expect(screen.getByText('30:00 / 2:00:00')).toBeInTheDocument();
  });

  it('handles many events without losing the playhead', () => {
    renderWorkspace(Array.from({ length: 40 }, (_, index) => index * 1.5));
    loadMetadata(60);
    advanceTo(45);
    expect(playheadPercent()).toBeCloseTo(75, 1);
  });

  it('tolerates a zero-length video', () => {
    renderWorkspace([]);
    loadMetadata(0);
    advanceTo(0);
    expect(playheadPercent()).toBe(0);
    expect(seekBar()).toBeDisabled();
  });
});
