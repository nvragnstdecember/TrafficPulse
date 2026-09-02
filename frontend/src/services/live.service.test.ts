import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { FakeSocket, makeSessionMessage, stubWebSocket } from '@/test/live-doubles';

import { LiveConnection } from './live.service';

/**
 * The live socket's own contract: what it sends, what it lets through, and what it
 * does when it is torn down.
 *
 * Separate from the hook's tests because this is the layer that would silently
 * break a session in ways React state could not reveal — a `close()` during
 * connect, or an unknown message type taken as a fatal error.
 */

function connection() {
  const onMessage = vi.fn();
  const onClose = vi.fn();
  const onOpen = vi.fn();
  const live = new LiveConnection(
    { onMessage, onClose, onOpen },
    'ws://localhost/api/live/ws',
  );
  return { live, onMessage, onClose, onOpen };
}

beforeEach(() => {
  stubWebSocket();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('LiveConnection', () => {
  it('opens with the session declaration once the socket is ready', () => {
    const { live, onOpen } = connection();
    live.open({ width: 640, height: 480 });

    const socket = FakeSocket.latest!;
    // Nothing is sent before the socket is open.
    expect(socket.sent).toHaveLength(0);
    socket.connect();

    expect(socket.sent[0]).toEqual({
      type: 'start',
      width: 640,
      height: 480,
      scene_hash: null,
    });
    expect(onOpen).toHaveBeenCalledOnce();
  });

  it('carries a calibrated camera’s scene revision', () => {
    const { live } = connection();
    live.open({ width: 640, height: 480, sceneHash: 'abc123' });
    FakeSocket.latest!.connect();
    expect(FakeSocket.latest!.sent[0]).toMatchObject({ scene_hash: 'abc123' });
  });

  it('delivers known messages and ignores unknown ones', () => {
    const { live, onMessage } = connection();
    live.open({ width: 640, height: 480 });
    const socket = FakeSocket.latest!;
    socket.connect();

    socket.deliver(makeSessionMessage());
    // A newer server sending a message this client does not know is not a failure.
    socket.deliver({ type: 'something-the-future-added' });
    socket.onmessage?.({ data: 'not json' } as MessageEvent);

    expect(onMessage).toHaveBeenCalledOnce();
    expect(onMessage.mock.calls[0][0]).toMatchObject({ type: 'session' });
  });

  it('drops frames sent before the socket is open rather than throwing', () => {
    const { live } = connection();
    live.open({ width: 640, height: 480 });
    const socket = FakeSocket.latest!;

    expect(() =>
      live.sendFrame({ sequence: 0, captureSeconds: 0, data: 'aaa' }),
    ).not.toThrow();
    expect(socket.frames).toHaveLength(0);

    socket.connect();
    live.sendFrame({ sequence: 1, captureSeconds: 0.5, data: 'bbb' });
    expect(socket.frames).toEqual([
      { type: 'frame', sequence: 1, capture_seconds: 0.5, data: 'bbb' },
    ]);
  });

  it('distinguishes a stop we asked for from a connection that dropped', () => {
    const { live, onClose } = connection();
    live.open({ width: 640, height: 480 });
    const socket = FakeSocket.latest!;
    socket.connect();

    socket.serverClose(1006);
    expect(onClose).toHaveBeenCalledWith({ clean: false, code: 1006 });

    const second = connection();
    second.live.open({ width: 640, height: 480 });
    const other = FakeSocket.latest!;
    other.connect();
    second.live.stop();
    other.serverClose(1000);
    expect(second.onClose).toHaveBeenCalledWith({ clean: true, code: 1000 });
  });

  it('closes safely while still connecting, and is idempotent', () => {
    const { live } = connection();
    live.open({ width: 640, height: 480 });
    const socket = FakeSocket.latest!;
    expect(socket.readyState).toBe(FakeSocket.CONNECTING);

    // An unmount during connect is the case this has to survive.
    live.close();
    expect(socket.closedWith).toEqual({ code: 1000, reason: 'client stopped monitoring' });
    expect(() => live.close()).not.toThrow();
  });

  it('reports whether it is open, so a caller cannot queue into a dead socket', () => {
    const { live } = connection();
    live.open({ width: 640, height: 480 });
    expect(live.isOpen).toBe(false);
    FakeSocket.latest!.connect();
    expect(live.isOpen).toBe(true);
    live.close();
    expect(live.isOpen).toBe(false);
  });
});
