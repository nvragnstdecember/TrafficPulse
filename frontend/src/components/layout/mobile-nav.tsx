import { Dialog, DialogContent, DialogTitle } from '../ui/dialog';
import { useUiStore } from '@/store/ui-store';

import { Brand, SidebarNav } from './sidebar-nav';

/** The mobile navigation drawer (a left sheet built on the Dialog primitive). */
export function MobileNav() {
  const open = useUiStore((state) => state.mobileSidebarOpen);
  const setOpen = useUiStore((state) => state.setMobileSidebarOpen);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="fixed inset-y-0 left-0 top-0 grid max-w-[16rem] translate-x-0 translate-y-0 grid-rows-[auto_1fr] gap-0 rounded-none border-r bg-sidebar p-0 data-[state=open]:animate-slide-in-right lg:hidden">
        <DialogTitle className="sr-only">Navigation</DialogTitle>
        <Brand />
        <SidebarNav onNavigate={() => setOpen(false)} />
      </DialogContent>
    </Dialog>
  );
}
