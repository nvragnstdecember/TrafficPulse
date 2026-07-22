import { Maximize, Minimize, Pause, Play, StepBack, StepForward } from 'lucide-react';

import { PLAYBACK_RATES } from '@/hooks/use-video-controller';
import { formatClock } from '@/lib/workspace';

import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { usePlayer } from './player-context';

/** The player's bottom control bar: transport, scrubber, speed, fullscreen. */
export function PlayerControls() {
  const { state, controls, isFullscreen } = usePlayer();
  const isPlaying = state.status === 'playing';
  const max = state.duration > 0 ? state.duration : 0;

  return (
    <div className="flex flex-col gap-1.5">
      <input
        type="range"
        min={0}
        max={max || 1}
        step={0.05}
        value={Math.min(state.currentTime, max || state.currentTime)}
        onChange={(event) => controls.seek(Number(event.target.value))}
        aria-label="Seek"
        disabled={max === 0}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
      />
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={controls.toggle}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => controls.stepFrame(-1)}
          aria-label="Previous frame"
        >
          <StepBack className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => controls.stepFrame(1)}
          aria-label="Next frame"
        >
          <StepForward className="size-4" />
        </Button>

        <span className="ml-1 select-none font-mono text-xs tabular-nums text-muted-foreground">
          {formatClock(state.currentTime)} / {formatClock(state.duration)}
        </span>

        <div className="ml-auto flex items-center gap-1">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" aria-label="Playback speed">
                {state.playbackRate}×
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {PLAYBACK_RATES.map((rate) => (
                <DropdownMenuCheckboxItem
                  key={rate}
                  checked={state.playbackRate === rate}
                  onCheckedChange={() => controls.setPlaybackRate(rate)}
                >
                  {rate}×
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            variant="ghost"
            size="icon"
            onClick={controls.toggleFullscreen}
            aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? <Minimize className="size-4" /> : <Maximize className="size-4" />}
          </Button>
        </div>
      </div>
    </div>
  );
}
