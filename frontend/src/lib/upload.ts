import { env } from './env';
import { formatBytes } from './format';

/**
 * Upload constraints + validation (H7C).
 *
 * Accepted formats and the size limit are configuration (from `env`), never
 * hardcoded at call sites. `validateUploadFile` is a pure function so the upload
 * workflow can validate before touching the network, and it is exhaustively
 * unit-testable.
 */
export interface UploadConstraints {
  maxBytes: number;
  acceptedExtensions: string[];
}

export const uploadConstraints: UploadConstraints = {
  maxBytes: env.maxUploadBytes,
  acceptedExtensions: env.acceptedVideoFormats,
};

export type UploadRejectionReason = 'empty' | 'unsupported-format' | 'too-large';

export interface UploadValidationOk {
  ok: true;
}
export interface UploadValidationError {
  ok: false;
  reason: UploadRejectionReason;
  message: string;
}
export type UploadValidationResult = UploadValidationOk | UploadValidationError;

/** The file extension (with leading dot, lower-cased) or empty string. */
export function fileExtension(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

/** Validate a picked file against the configured constraints (pure). */
export function validateUploadFile(
  file: File,
  constraints: UploadConstraints = uploadConstraints,
): UploadValidationResult {
  if (file.size <= 0) {
    return { ok: false, reason: 'empty', message: 'The selected file is empty.' };
  }
  const extension = fileExtension(file.name);
  if (!constraints.acceptedExtensions.includes(extension)) {
    return {
      ok: false,
      reason: 'unsupported-format',
      message: `Unsupported format. Accepted: ${constraints.acceptedExtensions.join(', ')}.`,
    };
  }
  if (file.size > constraints.maxBytes) {
    return {
      ok: false,
      reason: 'too-large',
      message: `File is too large (max ${formatBytes(constraints.maxBytes)}).`,
    };
  }
  return { ok: true };
}

/** The `accept` attribute value for a native file input. */
export function acceptAttribute(constraints: UploadConstraints = uploadConstraints): string {
  return [...constraints.acceptedExtensions, 'video/*'].join(',');
}
