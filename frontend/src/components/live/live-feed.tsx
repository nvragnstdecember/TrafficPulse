import { CameraOff, VideoOff } from 'lucide-react';

import { type CameraState, type MonitorState } from '@/lib/live';
import { cn } from '@/lib/utils';

export interface LiveFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  cameraState: CameraState;
  monitorState: MonitorState;
  /** The latest annotated frame, or null when nothing has come back yet. */
  annotatedSrc: string | null;
}

/**
 * The camera stage: a smooth local preview, with the analysed frame drawn over it.
 *
 * Two layers rather than one, and that is the point of the component. The `<video>`
 * runs at the camera's own rate and never stalls; the annotated still sits on top
 * and updates only when inference produces one, which on this hardware is a
 * fraction of that rate. Showing only the annotated frame would look broken (a
 * stuttering feed); showing only the preview would hide the system's actual output.
 * Layering them keeps the motion honest *and* the analysis visible, and the stats
 * strip underneath says which rate is which.
 *
 * While monitoring, the preview is dimmed under the analysed frame so a viewer can
 * tell at a glance which pixels the system has actually reasoned about.
 */
export function LiveFeed({ videoRef, cameraState, monitorState, annotatedSrc }: LiveFeedProps) {
  const monitoring = monitorState === 'monitoring' || monitorState === 'connecting';
  const showAnnotated = monitoring && annotatedSrc !== null;

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-lg border bg-black">
      <video
        ref={videoRef}
        muted
        playsInline
        autoPlay
        aria-label="Live camera preview"
        className={cn(
          'size-full object-contain transition-opacity',
          cameraState === 'ready' ? 'opacity-100' : 'opacity-0',
          showAnnotated && 'opacity-30',
        )}
      />

      {showAnnotated ? (
        <img
          src={annotatedSrc}
          alt="Latest analysed frame with detections, tracks and helmet state drawn on it"
          className="absolute inset-0 size-full object-contain"
        />
      ) : null}

      {cameraState !== 'ready' ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
          {cameraState === 'error' ? (
            <VideoOff className="size-8" aria-hidden="true" />
          ) : (
            <CameraOff className="size-8" aria-hidden="true" />
          )}
          <p className="text-sm">
            {cameraState === 'requesting'
              ? 'Waiting for camera permission…'
              : cameraState === 'error'
                ? 'The camera is not available'
                : 'Camera is off'}
          </p>
        </div>
      ) : null}

      {monitoring && !showAnnotated ? (
        <div className="absolute inset-x-0 bottom-0 bg-black/60 px-3 py-2 text-center text-xs text-white">
          {monitorState === 'connecting'
            ? 'Opening the monitoring session…'
            : 'Analysing — the first annotated frame appears when inference completes.'}
        </div>
      ) : null}
    </div>
  );
}
