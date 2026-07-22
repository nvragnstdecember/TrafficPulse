import { useNotificationsStore } from '@/store/notifications-store';

import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from './toast';

/**
 * Renders the notifications store as Radix toasts (H7B).
 *
 * The store is the single source of truth; Radix drives auto-dismiss via each
 * toast's `duration`, and `onOpenChange(false)` prunes it from the store.
 */
export function Toaster() {
  const notifications = useNotificationsStore((state) => state.notifications);
  const dismiss = useNotificationsStore((state) => state.dismiss);

  return (
    <ToastProvider swipeDirection="right">
      {notifications.map((notification) => (
        <Toast
          key={notification.id}
          variant={notification.variant}
          duration={notification.duration ?? Number.POSITIVE_INFINITY}
          onOpenChange={(open) => {
            if (!open) dismiss(notification.id);
          }}
        >
          <div className="flex flex-col gap-1">
            <ToastTitle>{notification.title}</ToastTitle>
            {notification.description ? (
              <ToastDescription>{notification.description}</ToastDescription>
            ) : null}
          </div>
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  );
}
