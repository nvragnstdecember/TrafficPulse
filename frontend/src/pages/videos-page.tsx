import { Video } from 'lucide-react';

import { PageHeader } from '@/components/common/page-header';
import { PlayerProvider } from '@/components/workspace/player-context';
import { UploadDropzone } from '@/components/workspace/upload-dropzone';
import { WorkspaceIntro } from '@/components/workspace/workspace-intro';
import { WorkspaceView } from '@/components/workspace/workspace-view';
import { useProcessing } from '@/hooks/use-processing';
import { overlayVideoSource } from '@/lib/overlay-source';
import { useUploadStore } from '@/store/upload-store';

/**
 * The video workspace (H7C): upload footage, watch processing live, then review
 * the confirmed violations against the video itself.
 *
 * The page only decides *which* stage to show. Until a video exists it renders
 * the dropzone — so no event query is issued for a video that isn't there — and
 * once one does the workspace mounts inside a single {@link PlayerProvider}, so
 * the player, timeline, and event panels share one playback controller.
 */
export default function VideosPage() {
  const processing = useProcessing();
  const objectUrl = useUploadStore((state) => state.objectUrl);
  const hasVideo = processing.video !== null || processing.phase === 'uploading';

  // Once the run has produced an annotated video, play that through the whole
  // workspace (player, timeline, evidence viewer all read this one source); until
  // then fall back to the original upload, which stays preserved separately.
  const displaySrc = overlayVideoSource(processing.job) ?? objectUrl;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Video}
        title="Video workspace"
        description="Upload source footage, follow detection live, and review every confirmed violation frame by frame."
      />

      {hasVideo ? (
        <PlayerProvider fps={processing.video?.fps ?? undefined}>
          <WorkspaceView processing={processing} objectUrl={displaySrc} />
        </PlayerProvider>
      ) : (
        <div className="space-y-6">
          <UploadDropzone
            onFileSelected={processing.actions.selectAndUpload}
            disabled={processing.isBusy}
            error={processing.error}
          />
          <WorkspaceIntro />
        </div>
      )}
    </div>
  );
}
