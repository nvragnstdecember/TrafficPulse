import {
  BarChart3,
  FileSearch,
  LayoutDashboard,
  type LucideIcon,
  Radio,
  Settings,
  Video,
} from 'lucide-react';

/** Canonical route paths — the single source used by the router and navigation. */
export const ROUTES = {
  dashboard: '/',
  videos: '/videos',
  live: '/live',
  evidence: '/evidence',
  analytics: '/analytics',
  settings: '/settings',
} as const;

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  /** Match the path exactly (used for the index route). */
  end?: boolean;
}

/** Primary navigation, rendered by both the desktop sidebar and mobile drawer. */
export const NAV_ITEMS: NavItem[] = [
  { path: ROUTES.dashboard, label: 'Dashboard', icon: LayoutDashboard, end: true },
  { path: ROUTES.videos, label: 'Videos', icon: Video },
  // Beside Videos, not under it: live monitoring is the other input mode, and it
  // runs the same pipeline over a camera rather than a stored file.
  { path: ROUTES.live, label: 'Live camera', icon: Radio },
  { path: ROUTES.evidence, label: 'Evidence', icon: FileSearch },
  { path: ROUTES.analytics, label: 'Analytics', icon: BarChart3 },
  { path: ROUTES.settings, label: 'Settings', icon: Settings },
];
