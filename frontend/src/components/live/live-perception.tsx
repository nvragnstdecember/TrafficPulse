import { Bike } from 'lucide-react';

import { type LiveMotorcycle, type LiveRider, type LiveTrack } from '@/api/types';
import { StatusChip } from '@/components/common/status-chip';
import { helmetLabel, helmetTone, occupancyLabel, occupancyTone } from '@/lib/live';

export interface LivePerceptionProps {
  tracks: LiveTrack[];
  motorcycles: LiveMotorcycle[];
  riders: LiveRider[];
  monitoring: boolean;
}

/**
 * What the system currently believes about the frame it last analysed.
 *
 * The annotated feed already draws all of this; this panel exists because a
 * drawing cannot be read aloud in a review and cannot say *why* something is
 * missing. In particular it is where the multi-rider case is stated in words:
 * "3 riders — DRIVER UNRESOLVED" is a finding, not a rendering artefact, and a
 * viewer must not have to infer it from the absence of a label on a box.
 *
 * Helmet state is shown as an observation and never as a verdict. A rider reading
 * `no helmet` here means the classifier read no helmet on that crop; whether that
 * is a violation is the reasoner's decision, and it appears in the event feed if
 * and only if the reasoner made it.
 */
export function LivePerception({
  tracks,
  motorcycles,
  riders,
  monitoring,
}: LivePerceptionProps) {
  const ridersByMotorcycle = new Map<string, LiveRider[]>();
  for (const rider of riders) {
    const bucket = ridersByMotorcycle.get(rider.motorcycle_track_id) ?? [];
    bucket.push(rider);
    ridersByMotorcycle.set(rider.motorcycle_track_id, bucket);
  }

  return (
    <div className="rounded-lg border bg-card">
      <div className="flex items-center justify-between border-b px-3 py-2">
        <h2 className="text-sm font-medium">Current frame</h2>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {tracks.length} track{tracks.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="space-y-3 p-3">
        {motorcycles.length === 0 ? (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Bike className="size-4 shrink-0" aria-hidden="true" />
            {monitoring
              ? 'No motorcycle with associated riders on the last analysed frame.'
              : 'Monitoring is not running.'}
          </p>
        ) : (
          <ul className="space-y-3">
            {motorcycles.map((motorcycle) => (
              <li key={motorcycle.motorcycle_track_id} className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs">
                    Track {motorcycle.motorcycle_track_id}
                  </span>
                  <StatusChip
                    tone={occupancyTone(motorcycle)}
                    label={occupancyLabel(motorcycle)}
                    dot={false}
                    className="text-2xs normal-case"
                  />
                </div>
                <ul className="space-y-1 pl-3">
                  {(ridersByMotorcycle.get(motorcycle.motorcycle_track_id) ?? []).map(
                    (rider) => (
                      <li
                        key={rider.rider_track_id}
                        className="flex items-center gap-2 text-xs text-muted-foreground"
                      >
                        <span className="font-mono">Rider {rider.rider_track_id}</span>
                        <StatusChip
                          tone={helmetTone(rider)}
                          label={helmetLabel(rider)}
                          dot={false}
                          className="text-2xs"
                        />
                        {rider.helmet_confidence !== null ? (
                          <span className="tabular-nums">
                            {(rider.helmet_confidence * 100).toFixed(0)}%
                          </span>
                        ) : null}
                      </li>
                    ),
                  )}
                </ul>
              </li>
            ))}
          </ul>
        )}

        {tracks.length > 0 ? (
          <p className="border-t pt-2 text-2xs text-muted-foreground">
            Detected on this frame:{' '}
            {[...new Set(tracks.map((track) => track.object_class))].join(', ')}
          </p>
        ) : null}
      </div>
    </div>
  );
}
