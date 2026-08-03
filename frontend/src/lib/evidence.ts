import { endpoints } from '@/api/endpoints';
import { type ArtifactReference, type EvidenceManifest } from '@/api/types';
import { env } from '@/lib/env';

/**
 * Evidence artifact selection (H14) — pure, no React, no network.
 *
 * The frontend does not decide which frames are evidence. Before H14 it did: the
 * viewer seeked the playing video to `trigger ± 1.5s` and called the result an
 * evidence frame, while the backend had independently picked the frames the engine
 * actually processed at its own configured margins. The two could disagree, and
 * nothing reconciled them.
 *
 * Now the manifest is the authority. A frame is evidence if, and only if, the
 * backend rendered it and the served manifest references it — which is exactly the
 * references carrying a `sha256`, since a content address is what a rendered
 * artifact has and a pre-render placeholder does not.
 */

/** The frame slots an evidence manifest can carry, in presentation order. */
export const EVIDENCE_FRAME_KINDS = ['before_frame', 'trigger_frame', 'after_frame'] as const;

export type EvidenceFrameKind = (typeof EVIDENCE_FRAME_KINDS)[number];

const FRAME_LABELS: Record<EvidenceFrameKind, string> = {
  before_frame: 'Before',
  trigger_frame: 'Trigger',
  after_frame: 'After',
};

export interface EvidenceFrame {
  kind: EvidenceFrameKind;
  label: string;
  /** The manifest reference this frame was resolved from. */
  artifact: ArtifactReference;
  /** Where the backend serves the rendered image. */
  url: string;
}

/**
 * Whether a reference points at bytes the backend actually rendered.
 *
 * A reference without a `sha256` names a source frame the engine picked but nobody
 * materialised (every manifest written before H14 is this shape). Requesting it
 * would 404, so it is not offered as a frame.
 */
export function isRenderedArtifact(
  artifact: ArtifactReference | null | undefined,
): artifact is ArtifactReference {
  return Boolean(artifact && artifact.sha256);
}

/** The URL the backend serves one rendered evidence frame from. */
export function evidenceArtifactUrl(eventId: string, kind: EvidenceFrameKind): string {
  return `${env.apiBaseUrl}${endpoints.evidenceArtifact(eventId, kind)}`;
}

/** The URL the complete evidence package downloads from. */
export function evidencePackageUrl(eventId: string): string {
  return `${env.apiBaseUrl}${endpoints.evidencePackage(eventId)}`;
}

/**
 * The rendered evidence frames for an event, in before → trigger → after order.
 *
 * Empty when nothing was rendered — an event from a pre-H14 repository, or one whose
 * render failed. The caller shows the metadata it does have rather than inventing
 * pixels, which is the whole point of the change.
 */
export function evidenceFrames(
  eventId: string,
  manifest: EvidenceManifest | undefined | null,
): EvidenceFrame[] {
  if (!manifest) return [];
  const frames: EvidenceFrame[] = [];
  for (const kind of EVIDENCE_FRAME_KINDS) {
    const artifact = manifest[kind];
    if (!isRenderedArtifact(artifact)) continue;
    frames.push({
      kind,
      label: FRAME_LABELS[kind],
      artifact,
      url: evidenceArtifactUrl(eventId, kind),
    });
  }
  return frames;
}

/** Every artifact reference on a manifest, for the viewer's artifact table. */
export function manifestArtifacts(
  manifest: EvidenceManifest | undefined | null,
): Array<{ label: string; artifact: ArtifactReference | null }> {
  if (!manifest) return [];
  return [
    { label: 'Before frame', artifact: manifest.before_frame },
    { label: 'Trigger frame', artifact: manifest.trigger_frame },
    { label: 'After frame', artifact: manifest.after_frame },
    { label: 'Clip', artifact: manifest.clip },
    { label: 'Trajectory', artifact: manifest.trajectory },
    { label: 'Plate crop', artifact: manifest.plate_crop },
  ];
}

/** A short, readable form of an artifact's content address (or a dash). */
export function shortDigest(artifact: ArtifactReference | null | undefined): string {
  if (!artifact?.sha256) return '—';
  return artifact.sha256.slice(0, 12);
}
