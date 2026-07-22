import { create } from 'zustand';

export type NotificationVariant = 'default' | 'success' | 'error' | 'warning';

export interface AppNotification {
  id: string;
  title: string;
  description?: string;
  variant: NotificationVariant;
  /** Auto-dismiss delay in ms; `null` keeps it until dismissed. */
  duration: number | null;
}

export interface NotifyInput {
  title: string;
  description?: string;
  variant?: NotificationVariant;
  duration?: number | null;
}

interface NotificationsState {
  notifications: AppNotification[];
  notify: (input: NotifyInput) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

let counter = 0;
function nextId(): string {
  counter += 1;
  return `ntf-${counter}`;
}

/**
 * App notifications (H7B) — the model behind the toast UI.
 *
 * A plain store (not a React context) so it can be pushed to from anywhere,
 * including non-component code like the global query error handler
 * (`useNotificationsStore.getState().notify(...)`). The `<Toaster/>` renders the
 * list; `duration` drives auto-dismiss there.
 */
export const useNotificationsStore = create<NotificationsState>((set) => ({
  notifications: [],
  notify: ({ title, description, variant = 'default', duration = 5000 }) => {
    const id = nextId();
    set((state) => ({
      notifications: [...state.notifications, { id, title, description, variant, duration }],
    }));
    return id;
  },
  dismiss: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((notification) => notification.id !== id),
    })),
  clear: () => set({ notifications: [] }),
}));

/** Push a notification from outside React (e.g. global error handlers). */
export function notify(input: NotifyInput): string {
  return useNotificationsStore.getState().notify(input);
}
