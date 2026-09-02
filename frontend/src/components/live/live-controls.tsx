import { Camera, CameraOff, Play, Square } from 'lucide-react';

import { StatusChip } from '@/components/common/status-chip';
import { Button } from '@/components/ui/button';
import {
  type CameraState,
  type LiveStatus,
  type MonitorState,
  liveStatusTone,
} from '@/lib/live';

export interface LiveControlsProps {
  cameraState: CameraState;
  monitorState: MonitorState;
  status: LiveStatus;
  /** Whether the server says a session could start at all right now. */
  serverReady: boolean;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onStartMonitoring: () => void;
  onStopMonitoring: () => void;
}

/**
 * The four controls, in the order the workflow uses them.
 *
 * Camera and monitoring are separate buttons because they are separate decisions:
 * opening a camera is a person granting access to their surroundings, and starting
 * monitoring is asking a server to analyse what that camera sees. Nothing here
 * starts inference as a side effect of the page loading, or of the camera turning
 * on — "Start monitoring" is always an explicit act.
 *
 * The status chip is the single fold of both lifecycles, so the header never shows
 * two competing states.
 */
export function LiveControls({
  cameraState,
  monitorState,
  status,
  serverReady,
  onStartCamera,
  onStopCamera,
  onStartMonitoring,
  onStopMonitoring,
}: LiveControlsProps) {
  const cameraOn = cameraState === 'ready';
  const monitoring = monitorState === 'monitoring' || monitorState === 'connecting';
  const busy = cameraState === 'requesting' || monitorState === 'stopping';

  return (
    <div className="flex flex-wrap items-center gap-2">
      <StatusChip tone={liveStatusTone(status)} label={status} />

      {cameraOn ? (
        <Button variant="outline" size="sm" onClick={onStopCamera} disabled={busy}>
          <CameraOff className="size-4" />
          Stop camera
        </Button>
      ) : (
        <Button size="sm" onClick={onStartCamera} disabled={busy}>
          <Camera className="size-4" />
          Start camera
        </Button>
      )}

      {monitoring ? (
        <Button variant="destructive" size="sm" onClick={onStopMonitoring} disabled={busy}>
          <Square className="size-4" />
          Stop monitoring
        </Button>
      ) : (
        <Button
          size="sm"
          onClick={onStartMonitoring}
          // Refused rather than allowed-then-failed: without a camera there is
          // nothing to send, and without a backend there is nothing to send it to.
          disabled={!cameraOn || !serverReady || busy}
          title={
            !cameraOn
              ? 'Start the camera first'
              : !serverReady
                ? 'This deployment cannot monitor a live camera right now'
                : undefined
          }
        >
          <Play className="size-4" />
          Start monitoring
        </Button>
      )}
    </div>
  );
}
