import { afterEach, describe, expect, it, vi } from 'vitest';

import { copyText } from './clipboard';

describe('copyText', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('writes to the clipboard and reports success', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    await expect(copyText('evt-1')).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith('evt-1');
  });

  it('returns false when the clipboard API is unavailable', async () => {
    vi.stubGlobal('navigator', {});
    await expect(copyText('x')).resolves.toBe(false);
  });

  it('returns false when the write rejects', async () => {
    vi.stubGlobal('navigator', {
      clipboard: { writeText: vi.fn(() => Promise.reject(new Error('denied'))) },
    });
    await expect(copyText('x')).resolves.toBe(false);
  });
});
