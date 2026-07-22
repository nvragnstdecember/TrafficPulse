import { Suspense } from 'react';
import { Outlet } from 'react-router-dom';

import { PageLoader } from '../common/page-loader';
import { MobileNav } from './mobile-nav';
import { Sidebar } from './sidebar';
import { StatusFooter } from './status-footer';
import { TopNav } from './top-nav';

/**
 * The responsive application shell (H7B): sidebar + mobile drawer, top bar,
 * scrollable main content (lazy pages behind a Suspense boundary), and the
 * status footer. This is the layout route every page renders inside.
 */
export function AppShell() {
  return (
    <div className="flex h-full w-full overflow-hidden">
      <a
        href="#main-content"
        className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground focus:not-sr-only focus:absolute focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <Sidebar />
      <MobileNav />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopNav />
        <main id="main-content" className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <Suspense fallback={<PageLoader />}>
              <Outlet />
            </Suspense>
          </div>
        </main>
        <StatusFooter />
      </div>
    </div>
  );
}
