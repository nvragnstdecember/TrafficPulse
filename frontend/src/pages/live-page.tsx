import { Radio } from 'lucide-react';

import { ErrorBanner } from '@/components/common/error-banner';
import { PageHeader } from '@/components/common/page-header';
import { LiveControls } from '@/components/live/live-controls';
import { LiveEventFeed } from '@/components/live/live-events';
import { LiveFeed } from '@/components/live/live-feed';
import { LivePerception } from '@/components/live/live-perception';
import { LiveStatsBar } from '@/components/live/live-stats';
import { SystemPostureStrip } from '@/components/workspace/system-posture';
import { useLiveCamera, useLiveReadiness } from '@/hooks/use-live-camera';
import { usePosture } from '@/hooks/use-system';

/**
 * Live camera monitoring.
 *
 * The same pipeline the video workspace runs, pointed at a camera instead of a
 * file: the frames go to a persistent backend session that keeps its tracker,
 * associations and reasoner history between them, and the violations that come
 * back are the ones the shipped reasoners confirmed. Nothing on this page decides
 * anything about a violation.
 *
 * Three honesty surfaces sit on this page on purpose, because a live feed is the
 * easiest place in the product to over-claim:
 *
 * * the **stats strip** publishes the measured inference rate beside the camera's
 *   own rate, and says in words that the two are not the same thing;
 * * the **event feed** lists the violations this camera is *not* being checked for,
 *   with the server's reason, so an empty feed cannot be misread as "all clear";
 * * the **capability strip** — the same one the workspace shows — states what this
 *   deployment's helmet reasoning is and is not entitled to claim.
 */
export default function LivePage() {
  const live = useLiveCamera();
  const readiness = useLiveReadiness();
  const posture = usePosture();

  const monitoring = live.monitorState === 'monitoring';
  const serverReady = readiness.data?.ready ?? false;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Radio}
        title="Live camera"
        description="Monitor a camera through the same detection, tracking, association and violation reasoning used for uploaded video."
        actions={
          <LiveControls
            cameraState={live.cameraState}
            monitorState={live.monitorState}
            status={live.status}
            serverReady={serverReady}
            onStartCamera={() => void live.startCamera()}
            onStopCamera={live.stopCamera}
            onStartMonitoring={live.startMonitoring}
            onStopMonitoring={live.stopMonitoring}
          />
        }
      />

      {live.error ? (
        <ErrorBanner
          title="Live monitoring problem"
          error={live.error}
          onRetry={live.dismissError}
        />
      ) : null}

      {readiness.data && !readiness.data.ready ? (
        <div
          role="status"
          className="rounded-md border border-warning/40 bg-warning/10 p-4 text-sm"
        >
          <p className="font-medium">Live monitoring is unavailable</p>
          <p className="text-muted-foreground">{readiness.data.detail}</p>
        </div>
      ) : null}

      {live.warnings.length > 0 ? (
        <div className="rounded-md border bg-card p-3 text-xs">
          <p className="mb-1 font-medium">Recent frame warnings</p>
          <ul className="space-y-0.5 text-muted-foreground">
            {live.warnings.map((warning, index) => (
              <li key={`${index}-${warning.slice(0, 24)}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <LiveFeed
            videoRef={live.videoRef}
            cameraState={live.cameraState}
            monitorState={live.monitorState}
            annotatedSrc={live.annotatedSrc}
          />
          <LiveStatsBar
            stats={live.stats}
            cameraFps={live.cameraFps}
            framesSent={live.framesSent}
          />
          <LivePerception
            tracks={live.tracks}
            motorcycles={live.motorcycles}
            riders={live.riders}
            monitoring={monitoring}
          />
        </div>

        <div className="flex min-h-0 flex-col gap-4 lg:max-h-[calc(100vh-14rem)]">
          <LiveEventFeed
            events={live.events}
            session={live.session}
            monitoring={monitoring}
          />
          <SystemPostureStrip posture={posture.data} />
        </div>
      </div>

      {live.session ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          Session {live.session.session_id} · camera {live.session.camera_id} ·{' '}
          {live.session.width}×{live.session.height}
          {live.session.scene_calibrated
            ? ' · reasoning through this camera’s calibrated scene'
            : ' · no calibrated scene, so only the geometry-free violations are evaluated'}
          . Camera frames and live events are held in memory for the length of this
          session and are never written to disk. Analysis restarts every{' '}
          {live.session.window_frames} processed frames to bound memory; track
          numbering restarts at that boundary.
        </p>
      ) : null}
    </div>
  );
}
