import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';

import { cn } from '@/lib/utils';
import { useUiStore } from '@/store/ui-store';

import { Button } from '../ui/button';
import { Separator } from '../ui/separator';
import { Brand, SidebarNav } from './sidebar-nav';

/** The persistent desktop navigation rail (hidden below the `lg` breakpoint). */
export function Sidebar() {
  const collapsed = useUiStore((state) => state.sidebarCollapsed);
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);

  return (
    <aside
      data-collapsed={collapsed}
      className={cn(
        'hidden shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground transition-[width] duration-slow lg:flex',
        collapsed ? 'w-16' : 'w-64',
      )}
    >
      <Brand collapsed={collapsed} />
      <Separator />
      <SidebarNav collapsed={collapsed} />
      <Separator />
      <div className={cn('p-3', collapsed && 'flex justify-center')}>
        <Button
          variant="ghost"
          size={collapsed ? 'icon' : 'sm'}
          onClick={toggleSidebar}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-pressed={collapsed}
          className={cn(!collapsed && 'w-full justify-start')}
        >
          {collapsed ? (
            <PanelLeftOpen className="size-4" />
          ) : (
            <>
              <PanelLeftClose className="size-4" />
              Collapse
            </>
          )}
        </Button>
      </div>
    </aside>
  );
}
