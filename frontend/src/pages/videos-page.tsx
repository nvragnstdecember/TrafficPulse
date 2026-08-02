import { ArrowLeft, Video } from 'lucide-react';

import { PageHeader } from '@/components/common/page-header';
import { Button } from '@/components/ui/button';
import { PlayerProvider } from '@/components/workspace/player-context';
import { UploadDropzone } from '@/components/workspace/upload-dropzone';
import { VideoLibrary } from '@/components/workspace/video-library';
import { WorkspaceIntro } from '@/components/workspace/workspace-intro';
import { WorkspaceView } from '@/components/workspace/workspace-view';
import { useProcessing } from '@/hooks/use-processing';
import { workspaceVideoSource } from '@/lib/video-source';
import { useUploadStore } from '@/store/upload-store';

/**
 * The video workspace (H7C; historical library H11): upload footage or reopen a
 * stored video, watch processing live, then review the confirmed violations against
 * the video itself.
 *
 * The page only decides *which* stage to show. Until a video is open it renders the
 * dropzone and the library — so no event query is issued for a video that isn't
 * there — and once one is, the workspace mounts inside a single {@link PlayerProvider},
 * so the player, timeline, and event panels share one playback controller. A video
 * opened from the library takes exactly the same path as one just uploaded.
 */
export default function VideosPage() {
  const processing = useProcessing();
  const objectUrl = useUploadStore((state) => state.objectUrl);
  const hasVideo = processing.video !== null || processing.phase === 'uploading';

  // One source for the whole workspace: the annotated video once it exists, else
  // the local file if this session picked it, else the stored upload streamed back
  // — which is what makes a video from the library playable at all.
  const displaySrc = workspaceVideoSource({
    job: processing.job,
    objectUrl,
    videoId: processing.video?.video_id,
  });

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Video}
        title="Video workspace"
        description="Upload source footage, follow detection live, and review every confirmed violation frame by frame."
        actions={
          hasVideo ? (
            <Button variant="outline" size="sm" onClick={processing.actions.remove}>
              <ArrowLeft className="size-4" />
              Video library
            </Button>
          ) : null
        }
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
          <VideoLibrary onOpen={processing.actions.openVideo} />
          <WorkspaceIntro />
        </div>
      )}
    </div>
  );
}
