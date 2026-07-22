import { type EventSort } from './types';

export interface EventListParams {
  videoId?: string;
  limit?: number;
  offset?: number;
  sort?: EventSort;
}

/**
 * Centralized, typed query keys (H7B).
 *
 * One factory per resource keeps cache keys consistent and makes targeted
 * invalidation unambiguous (e.g. `queryClient.invalidateQueries({ queryKey:
 * queryKeys.events.all })`).
 */
export const queryKeys = {
  health: ['health'] as const,
  metrics: ['metrics'] as const,
  jobs: {
    all: ['jobs'] as const,
    detail: (jobId: string) => ['jobs', jobId] as const,
  },
  events: {
    all: ['events'] as const,
    list: (params: EventListParams) => ['events', 'list', params] as const,
    detail: (eventId: string) => ['events', 'detail', eventId] as const,
  },
  evidence: {
    detail: (eventId: string) => ['evidence', eventId] as const,
  },
} as const;
