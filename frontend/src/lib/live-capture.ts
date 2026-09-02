/**
 * Turning the camera preview into frames the server can read.
 *
 * Isolated in its own module because it is the only part of live mode that needs
 * a real canvas: everything above it works on strings and numbers and is testable
 * in jsdom, and everything below it is the browser's own encoder.
 *
 * The capture size is fixed for the life of a session. The server builds its scene
 * from the size declared when the session opened and refuses a frame of any other
 * size — correctly, since a scene's geometry is measured in the frame it was drawn
 * on — so a capture that quietly followed a changing video resolution would produce
 * frames the session must reject.
 */

/** One captured frame: base64 JPEG, without the `data:` prefix. */
export type CapturedFrame = string | null;

export interface FrameCapture {
  readonly width: number;
  readonly height: number;
  /** Encode the video's current frame, or `null` if it is not drawable yet. */
  capture(): CapturedFrame;
  dispose(): void;
}

/** Read the camera's real frame size, or `null` before the stream has one. */
export function videoFrameSize(
  video: HTMLVideoElement,
): { width: number; height: number } | null {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return null;
  return { width, height };
}

/**
 * The camera's own negotiated frame rate, or `null` when it does not report one.
 *
 * `null` rather than a plausible default: this number is shown beside a measured
 * inference rate, and a made-up "30" would be the one figure on the panel that
 * nothing measured.
 */
export function streamFrameRate(stream: MediaStream): number | null {
  const track = stream.getVideoTracks()[0];
  const rate = track?.getSettings?.().frameRate;
  return typeof rate === 'number' && Number.isFinite(rate) ? rate : null;
}

/**
 * Build a capture bound to one video element at a fixed size.
 *
 * `quality` trades payload size against what the detector sees. It is deliberately
 * high: this is the image the model runs on, and compressing it hard to save
 * bandwidth on a localhost socket would degrade detection to buy nothing.
 */
export function createFrameCapture(
  video: HTMLVideoElement,
  size: { width: number; height: number },
  quality = 0.85,
): FrameCapture {
  const canvas = document.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext('2d');

  return {
    width: size.width,
    height: size.height,
    capture(): CapturedFrame {
      if (!context || video.readyState < 2) return null;
      try {
        context.drawImage(video, 0, 0, size.width, size.height);
        const url = canvas.toDataURL('image/jpeg', quality);
        const comma = url.indexOf(',');
        return comma === -1 ? null : url.slice(comma + 1);
      } catch {
        // A tainted or not-yet-ready canvas throws; one missed frame is not a
        // session failure, and the next tick tries again.
        return null;
      }
    },
    dispose(): void {
      canvas.width = 0;
      canvas.height = 0;
    },
  };
}
