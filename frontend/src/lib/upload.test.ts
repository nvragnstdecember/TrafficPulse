import { describe, expect, it } from 'vitest';

import { makeFile } from '@/test/fixtures';

import { acceptAttribute, fileExtension, uploadConstraints, validateUploadFile } from './upload';

const constraints = { maxBytes: 1000, acceptedExtensions: ['.mp4', '.mkv'] };

describe('fileExtension', () => {
  it('lower-cases the extension and tolerates missing ones', () => {
    expect(fileExtension('Clip.MP4')).toBe('.mp4');
    expect(fileExtension('archive.tar.gz')).toBe('.gz');
    expect(fileExtension('noextension')).toBe('');
  });
});

describe('validateUploadFile', () => {
  it('accepts a supported file within the size limit', () => {
    expect(validateUploadFile(makeFile('clip.mp4', 500), constraints)).toEqual({ ok: true });
  });

  it('rejects an empty file', () => {
    const result = validateUploadFile(makeFile('clip.mp4', 0), constraints);
    expect(result).toMatchObject({ ok: false, reason: 'empty' });
  });

  it('rejects an unsupported container', () => {
    const result = validateUploadFile(makeFile('clip.gif', 500), constraints);
    expect(result).toMatchObject({ ok: false, reason: 'unsupported-format' });
    expect(result.ok === false && result.message).toContain('.mp4');
  });

  it('rejects a file over the size limit', () => {
    const result = validateUploadFile(makeFile('clip.mp4', 1001), constraints);
    expect(result).toMatchObject({ ok: false, reason: 'too-large' });
  });

  it('defaults to the configured constraints', () => {
    expect(uploadConstraints.acceptedExtensions).toContain('.mp4');
    expect(validateUploadFile(makeFile('clip.mp4', 10))).toEqual({ ok: true });
  });
});

describe('acceptAttribute', () => {
  it('lists the extensions plus a generic video type', () => {
    expect(acceptAttribute(constraints)).toBe('.mp4,.mkv,video/*');
  });
});
