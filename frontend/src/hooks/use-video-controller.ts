import { useCallback, useEffect, useRef, useState } from 'react';

export type PlayerStatus = 'idle' | 'loading' | 'ready' | 'playing' | 'paused' | 'ended' | 'error';

export interface PlayerState {
  status: PlayerStatus;
  currentTime: number;
  duration: number;
  playbackRate: number;
  error: string | null;
}

export interface PlayerControls {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (time: number) => void;
  stepFrame: (direction: 1 | -1) => void;
  setPlaybackRate: (rate: number) => void;
  toggleFullscreen: () => void;
  retry: () => void;
}

export interface VideoController {
  setVideoEl: (el: HTMLVideoElement | null) => void;
  setContainerEl: (el: HTMLElement | null) => void;
  state: PlayerState;
  controls: PlayerControls;
  isFullscreen: boolean;
}

export const PLAYBACK_RATES = [0.5, 1, 1.5, 2] as const;

const INITIAL_STATE: PlayerState = {
  status: 'idle',
  currentTime: 0,
  duration: 0,
  playbackRate: 1,
  error: null,
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/**
 * Video player controller (H7C).
 *
 * Owns all playback state (status, time, duration, rate, error, fullscreen) and
 * the imperative controls, *outside* any UI component — a component only binds
 * the `<video>` element via `setVideoEl` and renders `state`/`controls`. Shared
 * across the player, its controls, and the timeline through a context provider.
 */
export function useVideoController(options?: { fps?: number }): VideoController {
  const fps = options?.fps && options.fps > 0 ? options.fps : 30;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLElement | null>(null);
  const detachRef = useRef<(() => void) | null>(null);
  const [state, setState] = useState<PlayerState>(INITIAL_STATE);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const patch = useCallback((next: Partial<PlayerState>) => {
    setState((prev) => ({ ...prev, ...next }));
  }, []);

  const setVideoEl = useCallback(
    (el: HTMLVideoElement | null) => {
      detachRef.current?.();
      detachRef.current = null;
      videoRef.current = el;
      if (!el) return;

      const onLoadStart = () => patch({ status: 'loading', error: null });
      const onLoadedMetadata = () =>
        patch({ status: 'ready', duration: Number.isFinite(el.duration) ? el.duration : 0 });
      const onDurationChange = () =>
        patch({ duration: Number.isFinite(el.duration) ? el.duration : 0 });
      const onTimeUpdate = () => patch({ currentTime: el.currentTime });
      const onPlay = () => patch({ status: 'playing' });
      const onPause = () => {
        if (!el.ended) patch({ status: 'paused' });
      };
      const onEnded = () => patch({ status: 'ended' });
      const onRateChange = () => patch({ playbackRate: el.playbackRate });
      const onWaiting = () => patch({ status: 'loading' });
      const onError = () => patch({ status: 'error', error: 'This video could not be played.' });

      el.addEventListener('loadstart', onLoadStart);
      el.addEventListener('loadedmetadata', onLoadedMetadata);
      el.addEventListener('durationchange', onDurationChange);
      el.addEventListener('timeupdate', onTimeUpdate);
      el.addEventListener('play', onPlay);
      el.addEventListener('pause', onPause);
      el.addEventListener('ended', onEnded);
      el.addEventListener('ratechange', onRateChange);
      el.addEventListener('waiting', onWaiting);
      el.addEventListener('error', onError);

      detachRef.current = () => {
        el.removeEventListener('loadstart', onLoadStart);
        el.removeEventListener('loadedmetadata', onLoadedMetadata);
        el.removeEventListener('durationchange', onDurationChange);
        el.removeEventListener('timeupdate', onTimeUpdate);
        el.removeEventListener('play', onPlay);
        el.removeEventListener('pause', onPause);
        el.removeEventListener('ended', onEnded);
        el.removeEventListener('ratechange', onRateChange);
        el.removeEventListener('waiting', onWaiting);
        el.removeEventListener('error', onError);
      };
    },
    [patch],
  );

  const setContainerEl = useCallback((el: HTMLElement | null) => {
    containerRef.current = el;
  }, []);

  const play = useCallback(() => {
    const promise = videoRef.current?.play?.();
    if (promise && typeof promise.catch === 'function') promise.catch(() => {});
  }, []);

  const pause = useCallback(() => {
    videoRef.current?.pause?.();
  }, []);

  const toggle = useCallback(() => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused || el.ended) play();
    else pause();
  }, [play, pause]);

  const seek = useCallback(
    (time: number) => {
      const el = videoRef.current;
      if (!el) return;
      const target = clamp(time, 0, el.duration || Number.MAX_SAFE_INTEGER);
      el.currentTime = target;
      patch({ currentTime: target });
    },
    [patch],
  );

  const stepFrame = useCallback(
    (direction: 1 | -1) => {
      const el = videoRef.current;
      if (!el) return;
      el.pause?.();
      seek(el.currentTime + direction * (1 / fps));
    },
    [fps, seek],
  );

  const setPlaybackRate = useCallback(
    (rate: number) => {
      const el = videoRef.current;
      if (el) el.playbackRate = rate;
      patch({ playbackRate: rate });
    },
    [patch],
  );

  const toggleFullscreen = useCallback(() => {
    const target = containerRef.current ?? videoRef.current;
    if (!target) return;
    if (typeof document !== 'undefined' && document.fullscreenElement) {
      document.exitFullscreen?.()?.catch?.(() => {});
    } else {
      target.requestFullscreen?.()?.catch?.(() => {});
    }
  }, []);

  const retry = useCallback(() => {
    const el = videoRef.current;
    if (!el) return;
    patch({ status: 'loading', error: null });
    el.load?.();
  }, [patch]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const onChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  useEffect(() => () => detachRef.current?.(), []);

  return {
    setVideoEl,
    setContainerEl,
    state,
    isFullscreen,
    controls: {
      play,
      pause,
      toggle,
      seek,
      stepFrame,
      setPlaybackRate,
      toggleFullscreen,
      retry,
    },
  };
}
