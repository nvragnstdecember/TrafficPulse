import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import {
  type LiveMotorcycle,
  type LiveResultMessage,
  type LiveRider,
  type LiveSessionMessage,
  type LiveStats,
  type LiveTrack,
} from '@/api/types';
import {
  type CameraState,
  type LiveEventRow,
  type MonitorState,
  cameraErrorMessage,
  cameraSupported,
  liveStatus,
  toEventRow,
} from '@/lib/live';
import { type FrameCapture, createFrameCapture, streamFrameRate, videoFrameSize } from '@/lib/live-capture';
import { type LiveConnection, liveService } from '@/services/live.service';

/**
 * How often the capture loop *considers* sending a frame. It is not a frame rate:
 * a tick sends only when the session has capacity, so the achieved rate settles at
 * whatever inference can absorb. A fast tick simply means the next frame goes out
 * promptly once a slot frees, rather than up to a fixed interval late.
 */
const CAPTURE_TICK_MS = 40;

/**
 * Frames allowed in flight at once.
 *
 * One would leave the server idle for a whole round trip after every result; three
 * would put a frame in the queue that is already stale by the time it is reached.
 * Two keeps the detector fed while bounding how old the oldest un-processed frame
 * can be to a single inference. The server enforces its own single-slot drop
 * behind this, so a client that ignored the limit could not grow a backlog anyway.
 */
const MAX_IN_FLIGHT = 2;

/** How many live events the feed keeps. Older ones scroll off; none is persisted. */
const MAX_EVENT_ROWS = 100;

/** Recent warnings shown under the feed, newest first. */
const MAX_WARNINGS = 5;

export interface LiveCameraController {
  cameraState: CameraState;
  monitorState: MonitorState;
  status: ReturnType<typeof liveStatus>;
  /** Attach to the preview `<video>`; the stream is bound here, not in the view. */
  videoRef: React.RefObject<HTMLVideoElement>;
  session: LiveSessionMessage | null;
  /** Data URL of the latest annotated frame, or null when nothing was drawn. */
  annotatedSrc: string | null;
  tracks: LiveTrack[];
  motorcycles: LiveMotorcycle[];
  riders: LiveRider[];
  events: LiveEventRow[];
  stats: LiveStats | null;
  /** The camera's own negotiated rate, or null when it does not report one. */
  cameraFps: number | null;
  /** Frames this client captured and sent (its share of the pipeline). */
  framesSent: number;
  error: string | null;
  warnings: string[];
  supported: boolean;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  startMonitoring: () => void;
  stopMonitoring: () => void;
  dismissError: () => void;
}

/**
 * The live camera controller: one camera, one optional monitoring session.
 *
 * Two lifecycles, kept apart on purpose. Opening the camera is a request to a
 * person for access to their surroundings and produces only a local preview;
 * starting monitoring is a request to a server to run inference on what that
 * camera sees. Merging them would mean a page that quietly began analysing the
 * moment it loaded, which is exactly the behaviour a live surveillance feature
 * must not have — so the camera never starts monitoring by itself, and stopping
 * monitoring leaves the preview running.
 *
 * Cleanup is unconditional. Every path out of this hook — stop, error, navigation,
 * unmount — stops the capture loop, closes the socket (which ends the backend
 * session) and stops every `MediaStream` track, so the camera indicator in the
 * browser goes out when the feature is no longer on screen.
 */
export function useLiveCamera(): LiveCameraController {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const connectionRef = useRef<LiveConnection | null>(null);
  const captureRef = useRef<FrameCapture | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inFlightRef = useRef(0);
  const sequenceRef = useRef(0);
  const captureStartRef = useRef(0);
  const annotatedRef = useRef<string | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>('off');
  const [monitorState, setMonitorState] = useState<MonitorState>('idle');
  const [session, setSession] = useState<LiveSessionMessage | null>(null);
  const [annotatedSrc, setAnnotatedSrc] = useState<string | null>(null);
  const [tracks, setTracks] = useState<LiveTrack[]>([]);
  const [motorcycles, setMotorcycles] = useState<LiveMotorcycle[]>([]);
  const [riders, setRiders] = useState<LiveRider[]>([]);
  const [events, setEvents] = useState<LiveEventRow[]>([]);
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [cameraFps, setCameraFps] = useState<number | null>(null);
  const [framesSent, setFramesSent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  const supported = useMemo(cameraSupported, []);

  const stopCaptureLoop = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    captureRef.current?.dispose();
    captureRef.current = null;
    inFlightRef.current = 0;
  }, []);

  const teardownSession = useCallback(() => {
    stopCaptureLoop();
    connectionRef.current?.close();
    connectionRef.current = null;
  }, [stopCaptureLoop]);

  const applyResult = useCallback((message: LiveResultMessage) => {
    inFlightRef.current = Math.max(0, inFlightRef.current - 1);
    setTracks(message.tracks);
    setMotorcycles(message.motorcycles);
    setRiders(message.riders);
    setStats(message.stats);
    if (message.annotated) {
      const next = `data:image/jpeg;base64,${message.annotated}`;
      annotatedRef.current = next;
      setAnnotatedSrc(next);
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (!supported) {
      setCameraState('error');
      setError(
        'This browser cannot open a camera. Camera capture needs a modern browser on a secure origin (https, or localhost).',
      );
      return;
    }
    setError(null);
    setCameraState('requesting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraFps(streamFrameRate(stream));
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        // Autoplay of a muted, playsinline preview is permitted; a rejection here
        // is not fatal (the user can press play), so it must not fail the camera.
        void video.play().catch(() => undefined);
      }
      setCameraState('ready');
    } catch (cause) {
      streamRef.current = null;
      setCameraState('error');
      setError(cameraErrorMessage(cause));
    }
  }, [supported]);

  const stopCamera = useCallback(() => {
    teardownSession();
    setMonitorState('idle');
    setSession(null);
    const stream = streamRef.current;
    streamRef.current = null;
    // Every track, individually: dropping the reference does not release the
    // device, and a camera left on after the user asked to stop it is the one
    // failure this feature must never have.
    stream?.getTracks().forEach((track) => track.stop());
    const video = videoRef.current;
    if (video) video.srcObject = null;
    setCameraState('off');
    setCameraFps(null);
  }, [teardownSession]);

  const startMonitoring = useCallback(() => {
    const video = videoRef.current;
    if (!video || cameraState !== 'ready') return;
    const size = videoFrameSize(video);
    if (!size) {
      setError('The camera has not produced a frame yet. Wait for the preview, then start monitoring.');
      return;
    }

    setError(null);
    setWarnings([]);
    setEvents([]);
    setStats(null);
    setFramesSent(0);
    sequenceRef.current = 0;
    inFlightRef.current = 0;
    captureStartRef.current = performance.now();
    setMonitorState('connecting');

    const capture = createFrameCapture(video, size);
    captureRef.current = capture;

    const connection = liveService.connect({
      onMessage: (message) => {
        switch (message.type) {
          case 'session':
            setSession(message);
            setMonitorState('monitoring');
            break;
          case 'result':
            applyResult(message);
            break;
          case 'events':
            setEvents((current) => {
              const received = new Date();
              const rows = message.events.map((event) => toEventRow(event, received));
              return [...rows.reverse(), ...current].slice(0, MAX_EVENT_ROWS);
            });
            break;
          case 'warning':
            // A warning means one frame was refused, so its in-flight slot frees
            // too -- otherwise a run of warnings would starve the capture loop.
            inFlightRef.current = Math.max(0, inFlightRef.current - 1);
            setWarnings((current) => [message.message, ...current].slice(0, MAX_WARNINGS));
            break;
          case 'stopped':
            setStats(message.stats);
            setMonitorState('idle');
            break;
          case 'error':
            setError(message.message);
            setMonitorState('error');
            stopCaptureLoop();
            break;
        }
      },
      onClose: ({ clean }) => {
        stopCaptureLoop();
        connectionRef.current = null;
        setSession(null);
        setMonitorState((current) => {
          if (current === 'error') return current;
          if (!clean) {
            setError(
              'The connection to the analysis server was lost. The camera is still running; start monitoring again to reconnect.',
            );
            return 'error';
          }
          return 'idle';
        });
      },
    });
    connectionRef.current = connection;
    connection.open({ width: size.width, height: size.height });

    timerRef.current = setInterval(() => {
      const active = connectionRef.current;
      const frames = captureRef.current;
      if (!active?.isOpen || !frames) return;
      if (inFlightRef.current >= MAX_IN_FLIGHT) return;
      const data = frames.capture();
      if (!data) return;
      inFlightRef.current += 1;
      active.sendFrame({
        sequence: sequenceRef.current++,
        // The client's own monotonic capture clock. The server stamps no
        // timestamp of its own; media time comes from the producer, as it does
        // from a file's PTS.
        captureSeconds: (performance.now() - captureStartRef.current) / 1000,
        data,
      });
      setFramesSent((count) => count + 1);
    }, CAPTURE_TICK_MS);
  }, [applyResult, cameraState, stopCaptureLoop]);

  const stopMonitoring = useCallback(() => {
    stopCaptureLoop();
    const connection = connectionRef.current;
    if (!connection) {
      setMonitorState('idle');
      return;
    }
    setMonitorState('stopping');
    connection.stop();
    // The server replies `stopped` and closes; both paths settle the state, and
    // `close()` here would race that reply.
  }, [stopCaptureLoop]);

  const dismissError = useCallback(() => setError(null), []);

  // Unmount: release the socket *and* the device. Navigating away from the page
  // must turn the camera light off.
  useEffect(() => {
    return () => {
      stopCaptureLoop();
      connectionRef.current?.close();
      connectionRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [stopCaptureLoop]);

  return {
    cameraState,
    monitorState,
    status: liveStatus(cameraState, monitorState),
    videoRef,
    session,
    annotatedSrc,
    tracks,
    motorcycles,
    riders,
    events,
    stats,
    cameraFps,
    framesSent,
    error,
    warnings,
    supported,
    startCamera,
    stopCamera,
    startMonitoring,
    stopMonitoring,
    dismissError,
  };
}

/**
 * The deployment's live readiness, asked before a camera is opened.
 *
 * Refetched while the page is open because the answer genuinely changes: session
 * slots are taken and released by other clients, and the pre-flight is only useful
 * if it reflects the server as it is now.
 */
export function useLiveReadiness() {
  return useQuery({
    queryKey: ['live', 'status'],
    queryFn: ({ signal }) => liveService.getStatus(signal),
    refetchInterval: 15_000,
    staleTime: 5_000,
  });
}
