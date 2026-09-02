import { act, renderHook, waitFor } from '@testing-library/react';
import { createRef } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  FakeSocket,
  FakeStream,
  FakeTrack,
  makeResultMessage,
  makeSessionMessage,
  stubCamera,
  stubCameraDenied,
  stubCanvasCapture,
  stubVideoFrames,
  stubWebSocket,
} from '@/test/live-doubles';
import { makeConfirmedEvent, mediaSeconds } from '@/test/fixtures';

import { useLiveCamera } from './use-live-camera';

/**
 * The live camera controller, driven against recording doubles.
 *
 * The hook's real capture loop, real in-flight accounting and real teardown run in
 * every test here; what is stubbed is only the three browser capabilities jsdom
 * lacks. So a passing test says the session behaves, not that a component renders.
 */

/** Attach the hook's video ref to a real element, as the page does. */
function mountController() {
  const video = document.createElement('video');
  document.body.appendChild(video);
  const rendered = renderHook(() => useLiveCamera());
  // The hook owns a ref object; the page assigns it via JSX. Doing it here is the
  // same assignment, and it is what makes the capture path reachable.
  (rendered.result.current.videoRef as React.MutableRefObject<HTMLVideoElement | null>).current =
    video;
  return { ...rendered, video };
}

async function startCameraAndMonitoring(rendered: ReturnType<typeof mountController>) {
  await act(async () => {
    await rendered.result.current.startCamera();
  });
  act(() => {
    rendered.result.current.startMonitoring();
  });
  const socket = FakeSocket.latest;
  if (!socket) throw new Error('no socket was opened');
  act(() => {
    socket.connect();
    socket.deliver(makeSessionMessage());
  });
  return socket;
}

beforeEach(() => {
  stubWebSocket();
  stubVideoFrames();
  stubCanvasCapture();
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('camera lifecycle', () => {
  it('does not open a session merely because the camera was started', async () => {
    // The load-bearing property of a live surveillance feature: turning on a
    // camera is not consent to analyse what it sees.
    stubCamera();
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });

    expect(rendered.result.current.cameraState).toBe('ready');
    expect(rendered.result.current.monitorState).toBe('idle');
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it('reports a denied permission with the recovery that applies', async () => {
    stubCameraDenied('NotAllowedError');
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });

    expect(rendered.result.current.cameraState).toBe('error');
    expect(rendered.result.current.error).toMatch(/allow camera access/i);
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it('reads the camera’s own negotiated frame rate rather than assuming one', async () => {
    stubCamera(new FakeStream([new FakeTrack({ frameRate: 24 })]));
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });

    expect(rendered.result.current.cameraFps).toBe(24);
  });

  it('reports no camera rate when the device does not declare one', async () => {
    stubCamera(new FakeStream([new FakeTrack({})]));
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });

    // Null, not 30: this number is shown beside a measured inference rate.
    expect(rendered.result.current.cameraFps).toBeNull();
  });

  it('releases every device track when the camera is stopped', async () => {
    const { stream } = stubCamera(new FakeStream([new FakeTrack(), new FakeTrack()]));
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });
    act(() => {
      rendered.result.current.stopCamera();
    });

    expect(rendered.result.current.cameraState).toBe('off');
    expect(stream.tracks.every((track) => track.stopped)).toBe(true);
  });
});

describe('monitoring lifecycle', () => {
  it('opens a session declaring the camera’s real frame size', async () => {
    stubCamera();
    stubVideoFrames(1280, 720);
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    expect(socket.sent[0]).toMatchObject({ type: 'start', width: 1280, height: 720 });
    expect(rendered.result.current.monitorState).toBe('monitoring');
    expect(rendered.result.current.session?.session_id).toBe('live-abc123');
  });

  it('streams frames into the open session', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(socket.frames.length).toBeGreaterThan(0);
    const frame = socket.frames[0];
    expect(frame.data).toBe('ZmFrZS1qcGVn');
    expect(frame.sequence).toBe(0);
    // The client supplies its own capture clock; the server invents no timestamp.
    expect(typeof frame.capture_seconds).toBe('number');
  });

  it('holds at most two frames in flight until results come back', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    // A long stretch of capture ticks with no result: the backlog must not grow.
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(socket.frames).toHaveLength(2);

    // One result frees exactly one slot.
    act(() => {
      socket.deliver(makeResultMessage({ sequence: 0 }));
    });
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(socket.frames).toHaveLength(3);
  });

  it('frees a slot when a frame is refused, so warnings cannot starve capture', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    expect(socket.frames).toHaveLength(2);

    act(() => {
      socket.deliver({ type: 'warning', code: 'live_frame_error', message: 'bad frame' });
    });
    await act(async () => {
      vi.advanceTimersByTime(500);
    });

    expect(socket.frames).toHaveLength(3);
    expect(rendered.result.current.warnings).toEqual(['bad frame']);
  });

  it('surfaces a processed frame’s perception and its measured metrics', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      socket.deliver(makeResultMessage());
    });

    expect(rendered.result.current.tracks).toHaveLength(1);
    expect(rendered.result.current.motorcycles[0].motorcycle_track_id).toBe('trk-1');
    expect(rendered.result.current.riders[0].helmet_label).toBe('no_helmet');
    expect(rendered.result.current.stats?.inference_fps).toBe(1.5);
    expect(rendered.result.current.annotatedSrc).toBe('data:image/jpeg;base64,YW5ub3RhdGVk');
  });

  it('keeps the previous annotated frame when a result has nothing to draw', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      socket.deliver(makeResultMessage());
      socket.deliver(makeResultMessage({ sequence: 1, annotated: null }));
    });

    // A frame the run published nothing for must not blank the stage.
    expect(rendered.result.current.annotatedSrc).toBe('data:image/jpeg;base64,YW5ub3RhdGVk');
  });

  it('shows newly confirmed events newest first', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      socket.deliver({
        type: 'events',
        events: [
          makeConfirmedEvent({ event_id: 'evt-1', violation_type: 'triple_riding' }),
          makeConfirmedEvent({
            event_id: 'evt-2',
            violation_type: 'wrong_way',
            start_at: mediaSeconds(4),
            trigger_at: mediaSeconds(6),
          }),
        ],
      });
    });

    const events = rendered.result.current.events;
    expect(events.map((event) => event.id)).toEqual(['evt-2', 'evt-1']);
    expect(events[0].observedSeconds).toBeCloseTo(2);
  });
});

describe('failure and cleanup', () => {
  it('ends the session on a server error and stops capturing', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    const before = socket.frames.length;

    act(() => {
      socket.deliver({
        type: 'error',
        code: 'live_inference_error',
        message: 'inference failed on a live frame',
      });
    });
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    expect(rendered.result.current.monitorState).toBe('error');
    expect(rendered.result.current.error).toMatch(/inference failed/);
    expect(socket.frames).toHaveLength(before);
    // The camera itself is untouched: a backend failure is not a camera failure.
    expect(rendered.result.current.cameraState).toBe('ready');
  });

  it('reports a dropped connection as a failure, not as a clean stop', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      socket.serverClose(1006);
    });

    await waitFor(() => expect(rendered.result.current.monitorState).toBe('error'));
    expect(rendered.result.current.error).toMatch(/connection to the analysis server was lost/i);
    expect(rendered.result.current.session).toBeNull();
  });

  it('asks the server to stop rather than dropping the socket', async () => {
    stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      rendered.result.current.stopMonitoring();
    });

    expect(socket.sent.at(-1)).toMatchObject({ type: 'stop' });
    expect(rendered.result.current.monitorState).toBe('stopping');

    act(() => {
      socket.deliver({
        type: 'stopped',
        session_id: 'live-abc123',
        stats: makeResultMessage().stats,
      });
    });
    expect(rendered.result.current.monitorState).toBe('idle');
    // Stopping analysis leaves the preview running, as designed.
    expect(rendered.result.current.cameraState).toBe('ready');
  });

  it('stopping the camera also ends the backend session', async () => {
    const { stream } = stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    act(() => {
      rendered.result.current.stopCamera();
    });

    expect(socket.closedWith).not.toBeNull();
    expect(stream.tracks.every((track) => track.stopped)).toBe(true);
    expect(rendered.result.current.monitorState).toBe('idle');
  });

  it('unmounting closes the socket and turns the camera off', async () => {
    const { stream } = stubCamera();
    const rendered = mountController();
    const socket = await startCameraAndMonitoring(rendered);

    rendered.unmount();

    // Navigating away must not leave an engine running or a camera light on.
    expect(socket.closedWith).not.toBeNull();
    expect(stream.tracks.every((track) => track.stopped)).toBe(true);
  });

  it('unmounting while the socket is still connecting is safe', async () => {
    stubCamera();
    const rendered = mountController();
    await act(async () => {
      await rendered.result.current.startCamera();
    });
    act(() => {
      rendered.result.current.startMonitoring();
    });

    const socket = FakeSocket.latest;
    expect(socket?.readyState).toBe(FakeSocket.CONNECTING);
    expect(() => rendered.unmount()).not.toThrow();
    expect(socket?.closedWith).not.toBeNull();
  });

  it('refuses to start monitoring before the camera has produced a frame', async () => {
    stubCamera();
    stubVideoFrames(0, 0);
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });
    act(() => {
      rendered.result.current.startMonitoring();
    });

    expect(FakeSocket.instances).toHaveLength(0);
    expect(rendered.result.current.error).toMatch(/has not produced a frame yet/i);
  });

  it('reports a browser with no camera API instead of failing on use', async () => {
    vi.stubGlobal('navigator', { ...globalThis.navigator, mediaDevices: undefined });
    const rendered = mountController();

    await act(async () => {
      await rendered.result.current.startCamera();
    });

    expect(rendered.result.current.cameraState).toBe('error');
    expect(rendered.result.current.error).toMatch(/cannot open a camera/i);
  });
});

describe('ref plumbing', () => {
  it('exposes a ref the page can attach to its preview element', () => {
    stubCamera();
    const rendered = renderHook(() => useLiveCamera());
    expect(rendered.result.current.videoRef).toEqual(createRef<HTMLVideoElement>());
  });
});
