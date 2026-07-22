import { lazy } from 'react';
import { type RouteObject, createBrowserRouter } from 'react-router-dom';

import { AppShell } from '@/components/layout/app-shell';

import { RouteError } from './route-error';
import { ROUTES } from './paths';

// Lazily-loaded pages → each becomes its own chunk (code splitting). The shell's
// Suspense boundary renders the page loader while a chunk resolves.
const DashboardPage = lazy(() => import('@/pages/dashboard-page'));
const VideosPage = lazy(() => import('@/pages/videos-page'));
const EvidencePage = lazy(() => import('@/pages/evidence-page'));
const AnalyticsPage = lazy(() => import('@/pages/analytics-page'));
const SettingsPage = lazy(() => import('@/pages/settings-page'));
const NotFoundPage = lazy(() => import('@/pages/not-found-page'));

/**
 * The application route tree. A single layout route renders the {@link AppShell};
 * every page is a child so it inherits the shell and its own `errorElement`, so a
 * page-level crash shows a recoverable error *inside* the shell rather than
 * blanking the app.
 */
export const routes: RouteObject[] = [
  {
    path: ROUTES.dashboard,
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <DashboardPage />, errorElement: <RouteError /> },
      { path: 'videos', element: <VideosPage />, errorElement: <RouteError /> },
      { path: 'evidence', element: <EvidencePage />, errorElement: <RouteError /> },
      { path: 'analytics', element: <AnalyticsPage />, errorElement: <RouteError /> },
      { path: 'settings', element: <SettingsPage />, errorElement: <RouteError /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

export function createRouter() {
  return createBrowserRouter(routes);
}
