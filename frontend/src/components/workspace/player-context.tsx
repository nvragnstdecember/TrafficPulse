import { createContext, useContext } from 'react';

import { type VideoController, useVideoController } from '@/hooks/use-video-controller';

const PlayerContext = createContext<VideoController | null>(null);

/**
 * Provides one shared video controller to the player, its controls, and the
 * timeline — so playback state lives outside those components and they stay in
 * sync without prop drilling.
 */
export function PlayerProvider({ fps, children }: { fps?: number; children: React.ReactNode }) {
  const controller = useVideoController({ fps });
  return <PlayerContext.Provider value={controller}>{children}</PlayerContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function usePlayer(): VideoController {
  const controller = useContext(PlayerContext);
  if (!controller) throw new Error('usePlayer must be used within a PlayerProvider');
  return controller;
}
