import { useEffect } from 'react';

import { type PlayerControls, type PlayerState } from './use-video-controller';

export interface PlayerShortcutOptions {
  state: PlayerState;
  controls: PlayerControls;
  /** Select the next/previous event in the current (filtered) list. */
  onNextEvent?: () => void;
  onPreviousEvent?: () => void;
  enabled?: boolean;
}

/** Seconds jumped by the arrow keys. */
export const SEEK_STEP_SECONDS = 5;

export interface ShortcutHint {
  keys: string;
  description: string;
}

/** The shortcut legend rendered in the workspace (kept next to the bindings). */
export const PLAYER_SHORTCUTS: ShortcutHint[] = [
  { keys: 'Space / K', description: 'Play or pause' },
  { keys: '← / →', description: `Seek ${SEEK_STEP_SECONDS}s` },
  { keys: ', / .', description: 'Step one frame' },
  { keys: 'J / L', description: 'Previous / next event' },
  { keys: 'F', description: 'Fullscreen' },
];

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

/**
 * Keyboard shortcuts for the workspace player (H7C).
 *
 * Bound at the document so the shortcuts work wherever focus sits — except while
 * the user is typing in a field (search, filters), where every key must reach the
 * input. Modifier combinations are left to the browser.
 */
export function usePlayerShortcuts({
  state,
  controls,
  onNextEvent,
  onPreviousEvent,
  enabled = true,
}: PlayerShortcutOptions): void {
  useEffect(() => {
    if (!enabled || typeof document === 'undefined') return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (isTypingTarget(event.target)) return;

      switch (event.key) {
        case ' ':
        case 'k':
        case 'K':
          event.preventDefault();
          controls.toggle();
          return;
        case 'ArrowLeft':
          event.preventDefault();
          controls.seek(state.currentTime - SEEK_STEP_SECONDS);
          return;
        case 'ArrowRight':
          event.preventDefault();
          controls.seek(state.currentTime + SEEK_STEP_SECONDS);
          return;
        case ',':
          event.preventDefault();
          controls.stepFrame(-1);
          return;
        case '.':
          event.preventDefault();
          controls.stepFrame(1);
          return;
        case 'j':
        case 'J':
          event.preventDefault();
          onPreviousEvent?.();
          return;
        case 'l':
        case 'L':
          event.preventDefault();
          onNextEvent?.();
          return;
        case 'f':
        case 'F':
          event.preventDefault();
          controls.toggleFullscreen();
          return;
        default:
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [enabled, controls, state.currentTime, onNextEvent, onPreviousEvent]);
}
