import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import {
  type LiveReadiness,
  type LiveServerMessage,
  type LiveSessionListResponse,
} from '@/api/types';
import { env } from '@/lib/env';
import { parseServerMessage, socketUrl } from '@/lib/live';

export interface LiveConnectionHandlers {
  onMessage: (message: LiveServerMessage) => void;
  /**
   * The socket closed. `clean` distinguishes a stop we asked for from a
   * connection that dropped — the UI must not report a deliberate stop as a
   * failure, and must not report a dropped connection as a clean stop.
   */
  onClose: (info: { clean: boolean; code: number }) => void;
  onOpen?: () => void;
}

export interface StartSessionOptions {
  width: number;
  height: number;
  /** A stored scene revision for a calibrated fixed camera; omitted otherwise. */
  sceneHash?: string | null;
}

/**
 * One live monitoring session's socket, with the protocol's shape enforced.
 *
 * The service layer owns the wire, as it does for HTTP: nothing above this file
 * builds a message object, and nothing below it knows about React. What it adds
 * over a bare `WebSocket` is the two things a caller would otherwise get wrong —
 * messages are narrowed to known types before they reach a handler (an unknown
 * type is ignored, not thrown), and {@link close} is idempotent and safe from a
 * component unmount at any point in the connection's life, including while it is
 * still opening.
 */
export class LiveConnection {
  private socket: WebSocket | null = null;
  private closedByUs = false;

  constructor(
    private readonly handlers: LiveConnectionHandlers,
    private readonly url: string = socketUrl(
      endpoints.liveSocket,
      env.apiBaseUrl,
      typeof window === 'undefined' ? '' : window.location.origin,
    ),
  ) {}

  /** Open the socket and send the opening `start` message once it is ready. */
  open(options: StartSessionOptions): void {
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.onopen = () => {
      this.send({
        type: 'start',
        width: options.width,
        height: options.height,
        scene_hash: options.sceneHash ?? null,
      });
      this.handlers.onOpen?.();
    };
    socket.onmessage = (event: MessageEvent) => {
      let payload: unknown;
      try {
        payload = JSON.parse(String(event.data));
      } catch {
        return; // Unparseable frames are ignored; the server never sends one.
      }
      const message = parseServerMessage(payload);
      if (message) this.handlers.onMessage(message);
    };
    socket.onclose = (event: CloseEvent) => {
      this.socket = null;
      this.handlers.onClose({ clean: this.closedByUs || event.code === 1000, code: event.code });
    };
    socket.onerror = () => {
      // A socket error is always followed by a close, which is where the UI is
      // told. Reporting both would show two failures for one event.
    };
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /** Send one captured frame. Silently ignored when the socket is not open. */
  sendFrame(frame: { sequence: number; captureSeconds: number; data: string }): void {
    this.send({
      type: 'frame',
      sequence: frame.sequence,
      capture_seconds: frame.captureSeconds,
      data: frame.data,
    });
  }

  /** Ask the server to end the session cleanly; it replies `stopped` and closes. */
  stop(): void {
    this.closedByUs = true;
    this.send({ type: 'stop' });
  }

  /**
   * Tear the connection down now. Idempotent, and safe to call from a cleanup
   * effect while the socket is still `CONNECTING` — closing a connecting socket is
   * legal and is exactly what an unmount during connect must do.
   */
  close(): void {
    this.closedByUs = true;
    const socket = this.socket;
    this.socket = null;
    if (!socket) return;
    socket.onmessage = null;
    socket.onopen = null;
    socket.onerror = null;
    socket.onclose = null;
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close(1000, 'client stopped monitoring');
    }
  }

  private send(payload: object): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify(payload));
  }
}

/**
 * Live camera service: the readiness pre-flight, the session listing, and the
 * socket factory. Hooks call these; components never do.
 */
export const liveService = {
  /** Whether live monitoring can start — asked before requesting camera access. */
  getStatus(signal?: AbortSignal): Promise<LiveReadiness> {
    return apiClient.get<LiveReadiness>(endpoints.liveStatus, { signal });
  },
  listSessions(signal?: AbortSignal): Promise<LiveSessionListResponse> {
    return apiClient.get<LiveSessionListResponse>(endpoints.liveSessions, { signal });
  },
  connect(handlers: LiveConnectionHandlers, url?: string): LiveConnection {
    return new LiveConnection(handlers, url);
  },
};
