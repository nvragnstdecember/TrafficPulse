import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { usePlayerShortcuts } from './use-player-shortcuts';
import { type PlayerControls, useVideoController } from './use-video-controller';

/**
 * A minimal stand-in for `<video>`: jsdom implements neither playback nor the
 * media clock, so the controller is exercised against a recording double.
 */
function createFakeVideo() {
  const listeners = new Map<string, Set<EventListener>>();
  const el = {
    duration: 60,
    currentTime: 0,
    playbackRate: 1,
    paused: true,
    ended: false,
    play: vi.fn(() => {
      el.paused = false;
      el.emit('play');
      return Promise.resolve();
    }),
    pause: vi.fn(() => {
      el.paused = true;
      el.emit('pause');
    }),
    load: vi.fn(),
    addEventListener: vi.fn((type: string, fn: EventListener) => {
      const set = listeners.get(type) ?? new Set<EventListener>();
      set.add(fn);
      listeners.set(type, set);
    }),
    removeEventListener: vi.fn((type: string, fn: EventListener) => {
      listeners.get(type)?.delete(fn);
    }),
    emit(type: string) {
      listeners.get(type)?.forEach((fn) => fn(new Event(type)));
    },
    listenerCount() {
      return [...listeners.values()].reduce((total, set) => total + set.size, 0);
    },
  };
  return el;
}

function mount(fps?: number) {
  const video = createFakeVideo();
  const hook = renderHook(() => useVideoController({ fps }));
  act(() => hook.result.current.setVideoEl(video as unknown as HTMLVideoElement));
  return { video, hook };
}

describe('useVideoController', () => {
  it('starts idle and tracks metadata, time, and playback events', () => {
    const { video, hook } = mount();
    expect(hook.result.current.state.status).toBe('idle');

    act(() => video.emit('loadstart'));
    expect(hook.result.current.state.status).toBe('loading');

    act(() => video.emit('loadedmetadata'));
    expect(hook.result.current.state).toMatchObject({ status: 'ready', duration: 60 });

    act(() => {
      video.currentTime = 12;
      video.emit('timeupdate');
    });
    expect(hook.result.current.state.currentTime).toBe(12);

    act(() => video.emit('play'));
    expect(hook.result.current.state.status).toBe('playing');
    act(() => video.emit('pause'));
    expect(hook.result.current.state.status).toBe('paused');
    act(() => video.emit('ended'));
    expect(hook.result.current.state.status).toBe('ended');
  });

  it('toggles playback against the element state', () => {
    const { video, hook } = mount();
    act(() => hook.result.current.controls.toggle());
    expect(video.play).toHaveBeenCalled();
    act(() => hook.result.current.controls.toggle());
    expect(video.pause).toHaveBeenCalled();
  });

  it('clamps seeks to the media duration', () => {
    const { video, hook } = mount();
    act(() => hook.result.current.controls.seek(90));
    expect(video.currentTime).toBe(60);
    act(() => hook.result.current.controls.seek(-5));
    expect(video.currentTime).toBe(0);
    expect(hook.result.current.state.currentTime).toBe(0);
  });

  it('steps a single frame at the configured fps and pauses first', () => {
    const { video, hook } = mount(25);
    act(() => {
      video.currentTime = 10;
      hook.result.current.controls.stepFrame(1);
    });
    expect(video.pause).toHaveBeenCalled();
    expect(video.currentTime).toBeCloseTo(10.04, 5);

    act(() => hook.result.current.controls.stepFrame(-1));
    expect(video.currentTime).toBeCloseTo(10, 5);
  });

  it('sets the playback rate on the element and in state', () => {
    const { video, hook } = mount();
    act(() => hook.result.current.controls.setPlaybackRate(1.5));
    expect(video.playbackRate).toBe(1.5);
    expect(hook.result.current.state.playbackRate).toBe(1.5);
  });

  it('surfaces media errors and recovers via retry', () => {
    const { video, hook } = mount();
    act(() => video.emit('error'));
    expect(hook.result.current.state).toMatchObject({ status: 'error' });
    expect(hook.result.current.state.error).toBeTruthy();

    act(() => hook.result.current.controls.retry());
    expect(video.load).toHaveBeenCalled();
    expect(hook.result.current.state).toMatchObject({ status: 'loading', error: null });
  });

  it('requests fullscreen on the container element', () => {
    const { hook } = mount();
    const container = { requestFullscreen: vi.fn(() => Promise.resolve()) };
    act(() => hook.result.current.setContainerEl(container as unknown as HTMLElement));
    act(() => hook.result.current.controls.toggleFullscreen());
    expect(container.requestFullscreen).toHaveBeenCalled();
  });

  it('detaches listeners when the element is replaced and on unmount', () => {
    const { video, hook } = mount();
    expect(video.listenerCount()).toBeGreaterThan(0);

    act(() => hook.result.current.setVideoEl(null));
    expect(video.listenerCount()).toBe(0);

    const second = createFakeVideo();
    act(() => hook.result.current.setVideoEl(second as unknown as HTMLVideoElement));
    hook.unmount();
    expect(second.listenerCount()).toBe(0);
  });

  it('ignores controls when no element is bound', () => {
    const hook = renderHook(() => useVideoController());
    expect(() => {
      act(() => {
        hook.result.current.controls.toggle();
        hook.result.current.controls.seek(5);
        hook.result.current.controls.stepFrame(1);
        hook.result.current.controls.retry();
        hook.result.current.controls.toggleFullscreen();
      });
    }).not.toThrow();
  });
});

describe('usePlayerShortcuts', () => {
  function setup(overrides: Partial<PlayerControls> = {}) {
    const controls: PlayerControls = {
      play: vi.fn(),
      pause: vi.fn(),
      toggle: vi.fn(),
      seek: vi.fn(),
      stepFrame: vi.fn(),
      setPlaybackRate: vi.fn(),
      toggleFullscreen: vi.fn(),
      retry: vi.fn(),
      ...overrides,
    };
    const onNextEvent = vi.fn();
    const onPreviousEvent = vi.fn();
    const hook = renderHook(
      (props: { enabled: boolean }) =>
        usePlayerShortcuts({
          state: {
            status: 'playing',
            currentTime: 20,
            duration: 60,
            playbackRate: 1,
            error: null,
          },
          controls,
          onNextEvent,
          onPreviousEvent,
          enabled: props.enabled,
        }),
      { initialProps: { enabled: true } },
    );
    return { controls, onNextEvent, onPreviousEvent, hook };
  }

  function press(key: string, init: KeyboardEventInit = {}) {
    act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }));
    });
  }

  it('binds transport, seek, frame-step, event, and fullscreen keys', () => {
    const { controls, onNextEvent, onPreviousEvent } = setup();

    press(' ');
    press('k');
    expect(controls.toggle).toHaveBeenCalledTimes(2);

    press('ArrowLeft');
    expect(controls.seek).toHaveBeenCalledWith(15);
    press('ArrowRight');
    expect(controls.seek).toHaveBeenCalledWith(25);

    press(',');
    expect(controls.stepFrame).toHaveBeenCalledWith(-1);
    press('.');
    expect(controls.stepFrame).toHaveBeenCalledWith(1);

    press('j');
    expect(onPreviousEvent).toHaveBeenCalled();
    press('l');
    expect(onNextEvent).toHaveBeenCalled();

    press('f');
    expect(controls.toggleFullscreen).toHaveBeenCalled();
  });

  it('ignores modifier combinations and unbound keys', () => {
    const { controls } = setup();
    press(' ', { ctrlKey: true });
    press('z');
    expect(controls.toggle).not.toHaveBeenCalled();
  });

  it('does not steal keys while the user is typing', () => {
    const { controls } = setup();
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    });
    expect(controls.toggle).not.toHaveBeenCalled();
    input.remove();
  });

  it('unbinds when disabled', () => {
    const { controls, hook } = setup();
    hook.rerender({ enabled: false });
    press(' ');
    expect(controls.toggle).not.toHaveBeenCalled();
  });
});
