import { act } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import * as storeBarrel from './index';
import { notify, useNotificationsStore } from './notifications-store';
import { useSelectionStore } from './selection-store';
import { useSettingsStore } from './settings-store';
import { useUiStore } from './ui-store';

describe('store barrel', () => {
  it('re-exports the public store surface', () => {
    expect(storeBarrel.useUiStore).toBeTypeOf('function');
    expect(storeBarrel.useSettingsStore).toBeTypeOf('function');
    expect(storeBarrel.useSelectionStore).toBeTypeOf('function');
    expect(storeBarrel.useNotificationsStore).toBeTypeOf('function');
    expect(storeBarrel.notify).toBeTypeOf('function');
  });
});

beforeEach(() => {
  localStorage.clear();
  useUiStore.setState({ sidebarCollapsed: false, mobileSidebarOpen: false });
  useSettingsStore.getState().reset();
  useSelectionStore.getState().clearSelection();
  useNotificationsStore.getState().clear();
});

describe('ui store', () => {
  it('toggles the sidebar and sets the mobile drawer', () => {
    act(() => useUiStore.getState().toggleSidebar());
    expect(useUiStore.getState().sidebarCollapsed).toBe(true);
    act(() => useUiStore.getState().setMobileSidebarOpen(true));
    expect(useUiStore.getState().mobileSidebarOpen).toBe(true);
  });
});

describe('settings store', () => {
  it('updates density and page size, and resets to defaults', () => {
    act(() => {
      useSettingsStore.getState().setDensity('compact');
      useSettingsStore.getState().setEventsPageSize(50);
    });
    expect(useSettingsStore.getState().density).toBe('compact');
    expect(useSettingsStore.getState().eventsPageSize).toBe(50);

    act(() => useSettingsStore.getState().reset());
    expect(useSettingsStore.getState().density).toBe('comfortable');
    expect(useSettingsStore.getState().eventsPageSize).toBe(25);
  });
});

describe('selection store', () => {
  it('selects and clears the current video', () => {
    act(() => useSelectionStore.getState().selectVideo('vid-1'));
    expect(useSelectionStore.getState().currentVideoId).toBe('vid-1');
    act(() => useSelectionStore.getState().clearSelection());
    expect(useSelectionStore.getState().currentVideoId).toBeNull();
  });
});

describe('notifications store', () => {
  it('adds, dismisses, and clears notifications', () => {
    let id = '';
    act(() => {
      id = useNotificationsStore.getState().notify({ title: 'Hello' });
    });
    expect(useNotificationsStore.getState().notifications).toHaveLength(1);
    expect(useNotificationsStore.getState().notifications[0]).toMatchObject({
      title: 'Hello',
      variant: 'default',
    });

    act(() => useNotificationsStore.getState().dismiss(id));
    expect(useNotificationsStore.getState().notifications).toHaveLength(0);
  });

  it('exposes a non-hook notify() helper', () => {
    act(() => {
      notify({ title: 'Oops', variant: 'error' });
    });
    expect(useNotificationsStore.getState().notifications[0].variant).toBe('error');
  });
});
