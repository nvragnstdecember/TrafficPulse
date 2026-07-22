import { AlertTriangle, FileVideo } from 'lucide-react';

import { Button } from '../ui/button';
import { Spinner } from '../ui/spinner';
import { PlayerControls } from './player-controls';
import { usePlayer } from './player-context';

export interface VideoPlayerProps {
  /** Object URL of the local file, or null when playback is unavailable. */
  src: string | null;
}

/**
 * The video surface: the `<video>` element, loading/error overlays, and the
 * control bar. All playback state comes from the shared player controller — this
 * component only binds the element and renders.
 */
export function VideoPlayer({ src }: VideoPlayerProps) {
  const { setVideoEl, setContainerEl, state, controls } = usePlayer();

  return (
    <div className="flex flex-col gap-2">
      <div
        ref={setContainerEl}
        className="relative aspect-video w-full overflow-hidden rounded-lg border bg-black"
      >
        {src ? (
          <video
            ref={setVideoEl}
            src={src}
            playsInline
            className="h-full w-full"
            onClick={controls.toggle}
            aria-label="Video preview"
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
            <FileVideo className="size-8" aria-hidden="true" />
            <p className="max-w-xs">
              Playback isn’t available for this session. Re-select the video file to preview it.
            </p>
          </div>
        )}

        {src && state.status === 'loading' ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <Spinner size={32} label="Loading video" />
          </div>
        ) : null}

        {src && state.status === 'error' ? (
          <div
            role="alert"
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70 p-4 text-center text-sm text-white"
          >
            <AlertTriangle className="size-8 text-destructive" aria-hidden="true" />
            <p>{state.error ?? 'This video could not be played.'}</p>
            <Button variant="outline" size="sm" onClick={controls.retry}>
              Try again
            </Button>
          </div>
        ) : null}
      </div>

      {src ? <PlayerControls /> : null}
    </div>
  );
}
