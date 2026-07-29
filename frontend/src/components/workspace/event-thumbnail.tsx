import { useEffect, useState } from 'react';

import { type ViolationType } from '@/api/types';
import { captureThumbnail } from '@/lib/thumbnail';
import { cn } from '@/lib/utils';
import { violationLabel } from '@/lib/workspace';

export interface EventThumbnailProps {
  /** Playable source for the processed video, or null when none is available. */
  src: string | null;
  /** Media-time second to sample. */
  seconds: number;
  violationType: ViolationType | string;
  className?: string;
}

/**
 * A still from the moment of the violation (Phase 2).
 *
 * Grabbed from the video the workspace already has (see `lib/thumbnail`), which is
 * why this needs no endpoint and no stored images. A frame is genuinely optional:
 * before the annotated video exists, on a source the canvas cannot read, and in any
 * environment without a real media stack, there simply is no still — so the
 * fallback is a first-class state, not an error. It shows the violation's initial
 * as a coloured tile, which keeps the card's rhythm and alignment identical
 * whether or not the image arrives.
 */
export function EventThumbnail({ src, seconds, violationType, className }: EventThumbnailProps) {
  const [frame, setFrame] = useState<string | null>(null);

  useEffect(() => {
    if (!src) {
      setFrame(null);
      return;
    }
    let cancelled = false;
    void captureThumbnail(src, seconds).then((value) => {
      if (!cancelled) setFrame(value);
    });
    return () => {
      cancelled = true;
    };
  }, [src, seconds]);

  const label = violationLabel(violationType);

  return (
    <span
      aria-hidden="true"
      className={cn(
        'flex size-12 shrink-0 items-center justify-center overflow-hidden rounded-md',
        'bg-muted text-sm font-semibold text-muted-foreground',
        className,
      )}
    >
      {frame ? (
        <img src={frame} alt="" className="size-full object-cover" loading="lazy" />
      ) : (
        label.charAt(0).toUpperCase()
      )}
    </span>
  );
}
