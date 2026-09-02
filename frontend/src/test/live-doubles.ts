import { vi } from 'vitest';

import { type LiveResultMessage, type LiveSessionMessage, type LiveStats } from '@/api/types';

/**
 * Doubles for the three browser capabilities live mode needs and jsdom does not
 * have: a camera, a canvas encoder, and a WebSocket.
 *
 * Each is a *recording* double rather than a mock of the code under test — the
 * hook's real capture loop, real back-pressure counter and real teardown all run,
 * and the tests assert on what these doubles observed. That is the difference
 * between proving the UI renders and proving the session behaves.
 */

/** A MediaStreamTrack double that records whether it was actually stopped. */
export class FakeTrack {
  stopped = false;
  kind = 'video';
  constructor(private readonly settings: MediaTrackSettings = { frameRate: 30 }) {}
  stop(): void {
    this.stopped = true;
  }
  getSettings(): MediaTrackSettings {
    return this.settings;
  }
}

/** A MediaStream double exposing its tracks so a test can assert they were released. */
export class FakeStream {
  constructor(public readonly tracks: FakeTrack[] = [new FakeTrack()]) {}
  getTracks(): FakeTrack[] {
    return this.tracks;
  }
  getVideoTracks(): FakeTrack[] {
    return this.tracks;
  }
}

export interface CameraStub {
  stream: FakeStream;
  getUserMedia: ReturnType<typeof vi.fn>;
}

/** Install a camera that grants access and yields `stream`. */
export function stubCamera(stream = new FakeStream()): CameraStub {
  const getUserMedia = vi.fn().mockResolvedValue(stream);
  vi.stubGlobal('navigator', {
    ...globalThis.navigator,
    mediaDevices: { getUserMedia },
  });
  return { stream, getUserMedia };
}

/** Install a camera that refuses, with a real `DOMException`-shaped rejection. */
export function stubCameraDenied(name = 'NotAllowedError'): ReturnType<typeof vi.fn> {
  const error = Object.assign(new Error('Permission denied'), { name });
  const getUserMedia = vi.fn().mockRejectedValue(error);
  vi.stubGlobal('navigator', {
    ...globalThis.navigator,
    mediaDevices: { getUserMedia },
  });
  return getUserMedia;
}

/**
 * Make every `<video>` report a frame size and be drawable.
 *
 * jsdom leaves `videoWidth`/`videoHeight` at 0 and has no media pipeline, so
 * without this the hook correctly refuses to start ("the camera has not produced a
 * frame yet") and no test of the session could run.
 */
export function stubVideoFrames(width = 640, height = 480): void {
  Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', {
    configurable: true,
    get: () => width,
  });
  Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', {
    configurable: true,
    get: () => height,
  });
  Object.defineProperty(HTMLVideoElement.prototype, 'readyState', {
    configurable: true,
    get: () => 4,
  });
}

/** Make the capture canvas encode to a fixed, recognisable JPEG payload. */
export function stubCanvasCapture(payload = 'ZmFrZS1qcGVn'): void {
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => ({ drawImage: vi.fn() }) as unknown as CanvasRenderingContext2D,
  ) as unknown as HTMLCanvasElement['getContext'];
  HTMLCanvasElement.prototype.toDataURL = vi.fn(() => `data:image/jpeg;base64,${payload}`);
}

interface SentMessage {
  type: string;
  [key: string]: unknown;
}

/**
 * A WebSocket double that records what the client sent and lets a test deliver
 * server messages on demand.
 *
 * Deliberately not auto-connecting: `open()` is called by the test, so the window
 * between `new WebSocket(...)` and the connection being established -- the window
 * an unmount has to survive -- is directly reachable.
 */
export class FakeSocket {
  static instances: FakeSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeSocket.CONNECTING;
  sent: SentMessage[] = [];
  closedWith: { code: number; reason: string } | null = null;

  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(payload: string): void {
    this.sent.push(JSON.parse(payload) as SentMessage);
  }

  close(code = 1000, reason = ''): void {
    this.readyState = FakeSocket.CLOSED;
    this.closedWith = { code, reason };
  }

  // --- test drivers ------------------------------------------------------------
  /** Complete the connection, delivering the client's `open` handler. */
  connect(): void {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  /** Deliver one server message. */
  deliver(message: object): void {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }

  /** Close from the server's side, cleanly or otherwise. */
  serverClose(code = 1000): void {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.({ code } as CloseEvent);
  }

  /** Every frame message this socket received from the client. */
  get frames(): SentMessage[] {
    return this.sent.filter((message) => message.type === 'frame');
  }

  static reset(): void {
    FakeSocket.instances = [];
  }

  static get latest(): FakeSocket | undefined {
    return FakeSocket.instances.at(-1);
  }
}

export function stubWebSocket(): typeof FakeSocket {
  FakeSocket.reset();
  vi.stubGlobal('WebSocket', FakeSocket);
  return FakeSocket;
}

// --- message builders --------------------------------------------------------------
export function makeLiveStats(overrides: Partial<LiveStats> = {}): LiveStats {
  return {
    frames_received: 1,
    frames_dropped: 0,
    frames_processed: 1,
    frames_rejected: 0,
    frames_out_of_order: 0,
    active_tracks: 2,
    events_emitted: 0,
    windows_completed: 0,
    window_frames_processed: 1,
    uptime_seconds: 3,
    inference_fps: 1.5,
    processing_ms_mean: 620,
    latency_ms_mean: 660,
    latency_ms_last: 640,
    ...overrides,
  };
}

export function makeSessionMessage(
  overrides: Partial<LiveSessionMessage> = {},
): LiveSessionMessage {
  return {
    type: 'session',
    session_id: 'live-abc123',
    camera_id: 'cam-live-abc123',
    width: 640,
    height: 480,
    scene_hash: null,
    scene_calibrated: false,
    running_violations: ['triple_riding'],
    unavailable_violations: [
      {
        violation_type: 'wrong_way',
        reason:
          "This camera's scene declares no legal travel direction, so there is nothing to judge a vehicle's heading against.",
      },
    ],
    window_frames: 600,
    ...overrides,
  };
}

export function makeResultMessage(
  overrides: Partial<LiveResultMessage> = {},
): LiveResultMessage {
  return {
    type: 'result',
    frame_index: 0,
    sequence: 0,
    capture_seconds: 0,
    tracks: [
      {
        track_id: 'trk-1',
        object_class: 'motorcycle',
        status: 'active',
        bbox: [10, 10, 60, 80],
        confidence: 0.91,
      },
    ],
    motorcycles: [{ motorcycle_track_id: 'trk-1', rider_count: 1, driver_resolved: true }],
    riders: [
      {
        rider_track_id: 'trk-2',
        motorcycle_track_id: 'trk-1',
        rider_count: 1,
        driver_resolved: true,
        helmet_label: 'no_helmet',
        helmet_confidence: 0.88,
        helmet_gated: false,
      },
    ],
    annotated: 'YW5ub3RhdGVk',
    window_rolled_over: false,
    stats: makeLiveStats(),
    ...overrides,
  };
}
