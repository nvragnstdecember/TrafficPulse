import { Link, NavLink } from 'react-router-dom';

import { Logo } from '@/assets/logo';
import { cn } from '@/lib/utils';
import { NAV_ITEMS, ROUTES } from '@/routes/paths';

export function Brand({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <Link
      to={ROUTES.dashboard}
      className="flex items-center gap-2.5 px-4 py-4"
      aria-label="TrafficPulse home"
    >
      <Logo className="size-7 shrink-0" />
      {!collapsed ? (
        <span className="text-[15px] font-semibold tracking-tight">TrafficPulse</span>
      ) : null}
    </Link>
  );
}

export interface SidebarNavProps {
  collapsed?: boolean;
  onNavigate?: () => void;
}

/** The primary navigation list, shared by the desktop rail and mobile drawer. */
export function SidebarNav({ collapsed = false, onNavigate }: SidebarNavProps) {
  return (
    <nav className="flex-1 space-y-1 px-3 py-2" aria-label="Primary">
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        return (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.end}
            onClick={onNavigate}
            title={collapsed ? item.label : undefined}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                collapsed && 'justify-center px-2',
                isActive
                  ? 'bg-sidebar-accent text-sidebar-foreground'
                  : 'text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              )
            }
          >
            <Icon className="size-[18px] shrink-0" aria-hidden="true" />
            <span className={cn('truncate', collapsed && 'sr-only')}>{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}
