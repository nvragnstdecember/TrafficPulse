/**
 * Clipboard helper (H7E).
 *
 * A single guarded entry point for copy-to-clipboard so components never touch
 * `navigator.clipboard` directly and every call is safe in jsdom / insecure
 * contexts (where the API is missing). Returns whether the copy succeeded so the
 * caller can show feedback.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (
      typeof navigator !== 'undefined' &&
      navigator.clipboard &&
      typeof navigator.clipboard.writeText === 'function'
    ) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the failure result */
  }
  return false;
}
