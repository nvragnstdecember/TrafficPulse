import { LogOut, Menu, Settings as SettingsIcon, UserRound } from 'lucide-react';
import { Link } from 'react-router-dom';

import { Logo } from '@/assets/logo';
import { APP_NAME } from '@/lib/app-info';
import { ROUTES } from '@/routes/paths';
import { useUiStore } from '@/store/ui-store';

import { ThemeToggle } from '../common/theme-toggle';
import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';

/** The top application bar: mobile menu trigger, brand, theme, and account menu. */
export function TopNav() {
  const setMobileOpen = useUiStore((state) => state.setMobileSidebarOpen);

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b bg-background/80 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/60 sm:px-4">
      <Button
        variant="ghost"
        size="icon"
        className="lg:hidden"
        aria-label="Open navigation"
        onClick={() => setMobileOpen(true)}
      >
        <Menu className="size-5" />
      </Button>

      <Link to={ROUTES.dashboard} className="flex items-center gap-2 lg:hidden">
        <Logo className="size-6" />
        <span className="text-sm font-semibold tracking-tight">{APP_NAME}</span>
      </Link>

      <div className="ml-auto flex items-center gap-1">
        <ThemeToggle />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="Account menu">
              <UserRound className="size-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link to={ROUTES.settings}>
                <SettingsIcon className="size-4" />
                Settings
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem disabled>
              <LogOut className="size-4" />
              Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
