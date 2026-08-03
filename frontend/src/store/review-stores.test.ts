import { act } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import { DEFAULT_EVENT_FILTERS } from '@/lib/workspace';

import { useSelectionStore } from './selection-store';
import { useWorkspacePrefsStore } from './workspace-prefs-store';

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useSelectionStore.getState().clearSelection();
    useWorkspacePrefsStore.setState({
      filters: DEFAULT_EVENT_FILTERS,
      sort: 'time-asc',
      selectionMode: false,
    });
  });
});

describe('workspace-prefs store (H7E)', () => {
  it('persists filters, sort, and selection mode', () => {
    act(() => {
      useWorkspacePrefsStore.getState().setFilters({ ...DEFAULT_EVENT_FILTERS, query: 'helmet' });
      useWorkspacePrefsStore.getState().setSort('severity-desc');
      useWorkspacePrefsStore.getState().setSelectionMode(true);
    });
    expect(useWorkspacePrefsStore.getState().filters.query).toBe('helmet');
    expect(useWorkspacePrefsStore.getState().sort).toBe('severity-desc');

    const persisted = JSON.parse(localStorage.getItem('trafficpulse-workspace-prefs') ?? '{}');
    expect(persisted.state).toMatchObject({
      sort: 'severity-desc',
      selectionMode: true,
      filters: { query: 'helmet' },
    });
  });

  it('rehydrates a pre-H9 blob into the current filter shape (H13.1)', async () => {
    // What a browser that last ran the H7C-era workspace left behind: filters
    // without the confidence ceiling, the time range, or `reviewStatuses`.
    // Zustand's default merge is shallow, so this object replaced the defaults
    // outright and the filters bar crashed reading `reviewStatuses.length`.
    localStorage.setItem(
      'trafficpulse-workspace-prefs',
      JSON.stringify({
        state: {
          filters: { query: 'helmet', violationTypes: ['no_helmet'], minConfidence: 0.4 },
          sort: 'time-asc',
          selectionMode: false,
        },
        version: 0,
      }),
    );

    await act(async () => {
      await useWorkspacePrefsStore.persist.rehydrate();
    });

    expect(useWorkspacePrefsStore.getState().filters).toEqual({
      ...DEFAULT_EVENT_FILTERS,
      query: 'helmet',
      violationTypes: ['no_helmet'],
      minConfidence: 0.4,
    });
  });

  it('falls back to defaults for corrupt persisted prefs (H13.1)', async () => {
    localStorage.setItem(
      'trafficpulse-workspace-prefs',
      JSON.stringify({ state: { filters: null, sort: 'by-vibes' }, version: 0 }),
    );

    await act(async () => {
      await useWorkspacePrefsStore.persist.rehydrate();
    });

    const state = useWorkspacePrefsStore.getState();
    expect(state.filters).toEqual(DEFAULT_EVENT_FILTERS);
    expect(state.sort).toBe('time-asc');
    expect(state.selectionMode).toBe(false);
    // The actions survive the merge — a normalized rehydration must not clobber
    // the store's own methods.
    expect(typeof state.setFilters).toBe('function');
  });

  it('resets filters without touching sort', () => {
    act(() => {
      useWorkspacePrefsStore.getState().setSort('severity-desc');
      useWorkspacePrefsStore
        .getState()
        .setFilters({ ...DEFAULT_EVENT_FILTERS, minConfidence: 0.5 });
      useWorkspacePrefsStore.getState().resetFilters();
    });
    expect(useWorkspacePrefsStore.getState().filters).toEqual(DEFAULT_EVENT_FILTERS);
    expect(useWorkspacePrefsStore.getState().sort).toBe('severity-desc');
  });
});

describe('selection store multi-select (H7E)', () => {
  it('toggles, sets, and clears the checked set', () => {
    act(() => useSelectionStore.getState().toggleChecked('a'));
    act(() => useSelectionStore.getState().toggleChecked('b'));
    expect([...useSelectionStore.getState().checkedEventIds].sort()).toEqual(['a', 'b']);

    act(() => useSelectionStore.getState().toggleChecked('a'));
    expect([...useSelectionStore.getState().checkedEventIds]).toEqual(['b']);

    act(() => useSelectionStore.getState().setChecked(['x', 'y', 'z']));
    expect(useSelectionStore.getState().checkedEventIds.size).toBe(3);

    act(() => useSelectionStore.getState().clearChecked());
    expect(useSelectionStore.getState().checkedEventIds.size).toBe(0);
  });

  it('clears the checked set when the whole selection is cleared', () => {
    act(() => {
      useSelectionStore.getState().toggleChecked('a');
      useSelectionStore.getState().selectEvent('a');
      useSelectionStore.getState().clearSelection();
    });
    expect(useSelectionStore.getState().checkedEventIds.size).toBe(0);
    expect(useSelectionStore.getState().selectedEventId).toBeNull();
  });
});
