import { type LiveStats } from '@/api/types';
import { cadenceSentence, formatLatency, formatRate, formatUptime } from '@/lib/live';

export interface LiveStatsBarProps {
  stats: LiveStats | null;
  /** The camera's negotiated rate, or null when it does not report one. */
  cameraFps: number | null;
  framesSent: number;
}

interface Reading {
  label: string;
  value: string;
  hint?: string;
}

/**
 * The operational strip: what is actually being measured, and one sentence saying
 * what the numbers mean together.
 *
 * The sentence is not decoration. A 30 fps preview beside a 1.5 fps inference rate
 * reads, to almost everyone, as "the AI is watching 30 frames a second" — and the
 * numbers alone do not correct that impression. The strip therefore states the
 * relationship in words, generated from the same two measured values, so it cannot
 * say something the readings do not.
 *
 * Nothing here is a target or a nominal figure. A value that has not been measured
 * yet renders as an em dash rather than a zero, because "0 fps" and "no
 * measurement yet" are different claims.
 */
export function LiveStatsBar({ stats, cameraFps, framesSent }: LiveStatsBarProps) {
  const readings: Reading[] = [
    {
      label: 'Camera',
      value: formatRate(cameraFps),
      hint: 'The rate the camera negotiated with the browser. The preview runs at this rate; inference does not.',
    },
    {
      label: 'AI inference',
      value: formatRate(stats?.inference_fps ?? null),
      hint: 'Throughput: frames per second the server actually completes, measured over the frames it has processed. Not the camera rate, and not one divided by the latency beside it.',
    },
    {
      label: 'Per frame',
      value: formatLatency(stats?.processing_ms_mean ?? null),
      hint: 'What one frame costs inside the pipeline: decode, detect, track, associate, classify, reason and draw.',
    },
    {
      label: 'Delay',
      value: formatLatency(stats?.latency_ms_last ?? null),
      hint: 'End-to-end for the most recent frame, including any wait behind a frame already being processed. This is the lag between the road and the screen.',
    },
    {
      label: 'Active tracks',
      value: stats ? String(stats.active_tracks) : '—',
      hint: 'Objects tracked on the most recently processed frame.',
    },
    {
      label: 'Sent / processed',
      value: stats ? `${framesSent} / ${stats.frames_processed}` : `${framesSent} / —`,
      hint: 'Frames this browser sent, and frames the server completed. A gap means frames are in flight.',
    },
    {
      label: 'Dropped',
      value: stats ? String(stats.frames_dropped) : '—',
      hint: 'Frames superseded by a newer one while inference was busy. Dropping is the design: a live view must show the road now, not a queue of old frames.',
    },
    {
      label: 'Session',
      value: stats ? formatUptime(stats.uptime_seconds) : '—',
      hint: 'How long this monitoring session has been open.',
    },
  ];

  return (
    <div className="space-y-2 rounded-lg border bg-card p-3">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4 lg:grid-cols-8">
        {readings.map((reading) => (
          <div key={reading.label} className="min-w-0" title={reading.hint}>
            <dt className="truncate text-2xs uppercase tracking-wide text-muted-foreground">
              {reading.label}
            </dt>
            <dd className="truncate font-mono text-sm tabular-nums">{reading.value}</dd>
          </div>
        ))}
      </dl>
      <p className="border-t pt-2 text-xs leading-relaxed text-muted-foreground">
        {cadenceSentence(cameraFps, stats?.inference_fps ?? null)}
      </p>
    </div>
  );
}
