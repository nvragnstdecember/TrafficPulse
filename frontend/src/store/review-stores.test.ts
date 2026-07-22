import { act } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import { DEFAULT_EVENT_FILTERS } from '@/lib/workspace';

import { useNotesStore } from './notes-store';
import { useSelectionStore } from './selection-store';
import { useWorkspacePrefsStore } from './workspace-prefs-store';

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useSelectionStore.getState().clearSelection();
    useNotesStore.setState({ notes: {} });
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

describe('notes store (H7E)', () => {
  it('sets, persists, and prunes blank notes', () => {
    act(() => useNotesStore.getState().setNote('evt-1', 'Confirmed on review'));
    expect(useNotesStore.getState().notes['evt-1']).toBe('Confirmed on review');

    const persisted = JSON.parse(localStorage.getItem('trafficpulse-notes') ?? '{}');
    expect(persisted.state.notes['evt-1']).toBe('Confirmed on review');

    act(() => useNotesStore.getState().setNote('evt-1', '   '));
    expect('evt-1' in useNotesStore.getState().notes).toBe(false);
  });

  it('clears a single note', () => {
    act(() => {
      useNotesStore.getState().setNote('evt-1', 'a');
      useNotesStore.getState().setNote('evt-2', 'b');
      useNotesStore.getState().clearNote('evt-1');
    });
    expect(useNotesStore.getState().notes).toEqual({ 'evt-2': 'b' });
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
