import {
  type ConfirmedEvent,
  type LiveMotorcycle,
  type LiveRider,
  type LiveServerMessage,
  type ViolationType,
} from '@/api/types';

import { type StatusTone } from '@/components/common/status-chip';
import { violationLabel } from './workspace';

/**
 * Pure logic for live camera monitoring (state naming, message narrowing, and the
 * readings shown beside the feed).
 *
 * Kept out of the hook and the components for the usual reason — it is the part
 * worth testing without a DOM — and out of the service because none of it touches
 * the network. Every function here is a pure function of a server message.
 */

/** The camera device's own lifecycle, independent of whether AI is running. */
export type CameraState = 'off' | 'requesting' | 'ready' | 'error';

/**
 * The monitoring session's lifecycle.
 *
 * Deliberately separate from {@link CameraState}: a camera can be previewing with
 * no session, and a session can fail with the camera still perfectly healthy. A
 * single merged status would have to lie about one of them.
 */
export type MonitorState = 'idle' | 'connecting' | 'monitoring' | 'stopping' | 'error';

/** The single label shown in the header, folded from the two states above. */
export type LiveStatus =
  | 'camera off'
  | 'requesting camera'
  | 'camera ready'
  | 'connecting'
  | 'monitoring'
  | 'stopping'
  | 'disconnected'
  | 'error';

export function liveStatus(camera: CameraState, monitor: MonitorState): LiveStatus {
  if (camera === 'error') return 'error';
  if (monitor === 'error') return 'error';
  if (camera === 'requesting') return 'requesting camera';
  if (camera === 'off') return monitor === 'idle' ? 'camera off' : 'disconnected';
  if (monitor === 'connecting') return 'connecting';
  if (monitor === 'monitoring') return 'monitoring';
  if (monitor === 'stopping') return 'stopping';
  return 'camera ready';
}

export function liveStatusTone(status: LiveStatus): StatusTone {
  switch (status) {
    case 'monitoring':
      return 'success';
    case 'connecting':
    case 'stopping':
    case 'requesting camera':
      return 'info';
    case 'camera ready':
      return 'info';
    case 'error':
      return 'error';
    case 'disconnected':
      return 'warning';
    case 'camera off':
      return 'neutral';
  }
}

/**
 * A person-readable reason for a `getUserMedia` failure.
 *
 * The browser's own messages are inconsistent across engines and unhelpful to a
 * non-specialist ("Requested device not found"), and the recovery differs per
 * cause — a denied permission needs a settings change, a busy device needs the
 * other application closed. So each case gets the sentence that actually says what
 * to do about it.
 */
export function cameraErrorMessage(error: unknown): string {
  const name = typeof error === 'object' && error !== null ? String((error as Error).name) : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was denied. Allow camera access for this site in your browser’s address bar or site settings, then start the camera again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera was found. Connect a camera (or select a different one in your browser’s site settings) and try again.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The camera could not be started — another application is probably using it. Close the other application and try again.';
    default:
      return typeof error === 'object' && error !== null && 'message' in error
        ? `The camera could not be started: ${String((error as Error).message)}`
        : 'The camera could not be started.';
  }
}

/** Whether this browser can capture a camera at all (secure context + API). */
export function cameraSupported(): boolean {
  return typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia);
}

/** Absolute ws:// or wss:// URL for a same-origin (or configured) API path. */
export function socketUrl(path: string, baseUrl: string, origin: string): string {
  const base = baseUrl || origin;
  const absolute = base.startsWith('http') ? base : `${origin}${base}`;
  return `${absolute.replace(/^http/, 'ws').replace(/\/$/, '')}${path}`;
}

/**
 * Narrow one parsed socket payload to a known server message, or `null`.
 *
 * Returning `null` rather than throwing is deliberate: an unknown `type` means the
 * server is newer than this client, which is a message to ignore, not a session to
 * kill.
 */
export function parseServerMessage(raw: unknown): LiveServerMessage | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const type = (raw as { type?: unknown }).type;
  if (
    type === 'session' ||
    type === 'result' ||
    type === 'events' ||
    type === 'warning' ||
    type === 'error' ||
    type === 'stopped'
  ) {
    return raw as LiveServerMessage;
  }
  return null;
}

/** One live event as the feed renders it. */
export interface LiveEventRow {
  id: string;
  violationType: ViolationType;
  label: string;
  /** Wall-clock time this client received it — a live feed is read as "just now". */
  receivedAt: Date;
  trackIds: string[];
  ruleId: string;
  /** Media seconds the violation was sustained before it triggered. */
  observedSeconds: number;
}

export function toEventRow(event: ConfirmedEvent, receivedAt: Date): LiveEventRow {
  const start = Date.parse(event.start_at);
  const trigger = Date.parse(event.trigger_at);
  return {
    id: event.event_id,
    violationType: event.violation_type,
    label: violationLabel(event.violation_type),
    receivedAt,
    trackIds: event.track_ids,
    ruleId: event.rule_id,
    observedSeconds:
      Number.isFinite(start) && Number.isFinite(trigger) ? (trigger - start) / 1000 : 0,
  };
}

/** `HH:MM:SS` in the viewer's own locale — a live feed is read against a wall clock. */
export function formatWallClock(at: Date): string {
  return at.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

/**
 * The one-line reading for a motorcycle's occupancy.
 *
 * The unresolved case is stated in full rather than abbreviated to a symbol,
 * because it is the finding: this system cannot say which of two riders is
 * driving, so a helmet reading on a multi-rider motorcycle cannot be attributed to
 * the driver at all.
 */
export function occupancyLabel(motorcycle: LiveMotorcycle): string {
  if (motorcycle.rider_count <= 1) return '1 rider';
  return `${motorcycle.rider_count} riders — DRIVER UNRESOLVED`;
}

export function occupancyTone(motorcycle: LiveMotorcycle): StatusTone {
  return motorcycle.driver_resolved ? 'neutral' : 'warning';
}

/** What the helmet classifier read for one rider, or why it read nothing. */
export function helmetLabel(rider: LiveRider): string {
  if (rider.helmet_gated) return 'crop refused';
  if (!rider.helmet_label) return 'not classified';
  return rider.helmet_label.replace('_', ' ');
}

export function helmetTone(rider: LiveRider): StatusTone {
  if (rider.helmet_gated || !rider.helmet_label) return 'neutral';
  if (rider.helmet_label === 'no_helmet') return 'warning';
  if (rider.helmet_label === 'helmet') return 'success';
  return 'info';
}

/** A measured rate, or an em dash. Never a placeholder number. */
export function formatRate(value: number | null | undefined, unit = 'fps'): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${value.toFixed(value < 10 ? 1 : 0)} ${unit}`;
}

export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export function formatUptime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

/**
 * The sentence under the frame-rate readings.
 *
 * The whole reason this exists: a camera preview at 30 fps beside an inference
 * rate of 1–2 fps invites the reading "the AI is analysing 30 frames a second".
 * The numbers alone do not correct that; a sentence does. It is generated from the
 * two measured rates, so it cannot drift from them.
 */
export function cadenceSentence(
  cameraFps: number | null,
  inferenceFps: number | null,
): string {
  if (inferenceFps === null) {
    return 'AI inference has not processed a frame yet, so no rate has been measured.';
  }
  const inference = inferenceFps.toFixed(1);
  if (cameraFps === null) {
    return `AI inference is processing about ${inference} frames per second. The preview is not analysed frame-for-frame.`;
  }
  return `The preview runs at about ${cameraFps.toFixed(0)} fps; AI inference processes about ${inference} of those frames per second. Frames captured while inference is busy are dropped, never queued.`;
}
