import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import {
  type ExpectationComparison,
  type ExpectationDeclaration,
  type ExpectationRecord,
} from '@/api/types';

/**
 * Controlled-demonstration service: declared expectations, and their comparison.
 *
 * The ground-truth side of a controlled demo. Everything here is about what a clip
 * was *declared* to contain; nothing here can produce, name or influence an event.
 * The detection side is read through `events.service` exactly as it always was, and
 * the comparison the backend returns is the only place the two meet.
 *
 * Kept apart from `scenes.service` deliberately. A scene is camera geometry that
 * the reasoners genuinely consume; an expectation is a demonstration's own record
 * that they must never see, and putting both behind one service would make that
 * distinction a comment rather than a boundary.
 */
export const demoService = {
  /** The video's declaration; 404s when nothing has been declared for it. */
  getExpectation(videoId: string, signal?: AbortSignal): Promise<ExpectationRecord> {
    return apiClient.get<ExpectationRecord>(endpoints.videoExpectation(videoId), { signal });
  },

  /**
   * Declare what this controlled clip was built to contain.
   *
   * Replacement, not merge: the stored declaration becomes exactly what is sent.
   * Declaring changes nothing about how the video is processed.
   */
  declare(
    videoId: string,
    declaration: ExpectationDeclaration,
    signal?: AbortSignal,
  ): Promise<ExpectationRecord> {
    return apiClient.put<ExpectationRecord>(
      endpoints.videoExpectation(videoId),
      declaration,
      { signal },
    );
  },

  /** Withdraw the declaration. Idempotent, and never touches confirmed events. */
  withdraw(videoId: string, signal?: AbortSignal): Promise<void> {
    return apiClient.delete<void>(endpoints.videoExpectation(videoId), { signal });
  },

  /**
   * Expected beside detected, for one run.
   *
   * `jobId` scopes the detected side to a single run, which is what a demonstration
   * wants — a video reprocessed with different rules has a different answer. A video
   * with no declaration is not an error: every detected family comes back
   * `unexpected`, which is the honest reading of "nothing was claimed here".
   */
  comparison(
    videoId: string,
    jobId?: string,
    signal?: AbortSignal,
  ): Promise<ExpectationComparison> {
    return apiClient.get<ExpectationComparison>(
      endpoints.videoExpectationComparison(videoId),
      { signal, query: { job_id: jobId } },
    );
  },
};
