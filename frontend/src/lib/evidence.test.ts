import { describe, expect, it } from 'vitest';

import { makeEvidence, makeRenderedEvidence } from '@/test/fixtures';

import {
  EVIDENCE_FRAME_KINDS,
  evidenceArtifactUrl,
  evidenceFrames,
  evidencePackageUrl,
  isRenderedArtifact,
  manifestArtifacts,
  shortDigest,
} from './evidence';

describe('rendered-artifact detection (H14)', () => {
  it('treats a content hash as the mark of a rendered artifact', () => {
    const rendered = makeRenderedEvidence();
    expect(isRenderedArtifact(rendered.trigger_frame)).toBe(true);
  });

  it('rejects a pre-render reference, which has no hash', () => {
    // The H11–H13 shape: a locator naming a frame nobody materialised.
    expect(isRenderedArtifact(makeEvidence().before_frame)).toBe(false);
    expect(isRenderedArtifact(null)).toBe(false);
    expect(isRenderedArtifact(undefined)).toBe(false);
  });
});

describe('evidence frame selection', () => {
  it('returns the rendered frames in before → trigger → after order', () => {
    const frames = evidenceFrames('evt-1', makeRenderedEvidence());
    expect(frames.map((frame) => frame.kind)).toEqual([...EVIDENCE_FRAME_KINDS]);
    expect(frames.map((frame) => frame.label)).toEqual(['Before', 'Trigger', 'After']);
  });

  it('points each frame at the backend artifact endpoint', () => {
    const [before] = evidenceFrames('evt-1', makeRenderedEvidence());
    expect(before.url).toBe('/api/evidence/evt-1/artifacts/before_frame');
  });

  it('omits a slot the backend did not render', () => {
    const frames = evidenceFrames('evt-1', makeRenderedEvidence({ before_frame: null }));
    expect(frames.map((frame) => frame.kind)).toEqual(['trigger_frame', 'after_frame']);
  });

  it('returns nothing for a pre-H14 manifest rather than inferring frames', () => {
    // The regression this replaced: the viewer used to invent `trigger ± 1.5s`.
    expect(evidenceFrames('evt-1', makeEvidence())).toEqual([]);
  });

  it('returns nothing when there is no manifest at all', () => {
    expect(evidenceFrames('evt-1', undefined)).toEqual([]);
    expect(evidenceFrames('evt-1', null)).toEqual([]);
  });

  it('encodes ids into the urls it builds', () => {
    expect(evidenceArtifactUrl('evt/1', 'trigger_frame')).toContain('evt%2F1');
    expect(evidencePackageUrl('evt/1')).toContain('evt%2F1');
  });
});

describe('manifest artifact presentation', () => {
  it('lists every typed slot, present or not', () => {
    const rows = manifestArtifacts(makeRenderedEvidence());
    expect(rows.map((row) => row.label)).toEqual([
      'Before frame',
      'Trigger frame',
      'After frame',
      'Clip',
      'Trajectory',
      'Plate crop',
    ]);
    expect(rows.find((row) => row.label === 'Clip')?.artifact).toBeNull();
  });

  it('is empty without a manifest', () => {
    expect(manifestArtifacts(undefined)).toEqual([]);
  });

  it('abbreviates a content hash and dashes an unrendered reference', () => {
    expect(shortDigest(makeRenderedEvidence().trigger_frame)).toBe('b'.repeat(12));
    expect(shortDigest(makeEvidence().before_frame)).toBe('—');
    expect(shortDigest(null)).toBe('—');
  });
});
