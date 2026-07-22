import { afterEach, describe, expect, it, vi } from 'vitest';

import { makeWorkspaceEvent, mediaSeconds } from '@/test/fixtures';

import {
  EVENT_CSV_COLUMNS,
  downloadTextFile,
  eventCsvRow,
  eventsToCsv,
  eventsToJsonModel,
  exportFilename,
  jsonString,
} from './export';

describe('eventsToCsv', () => {
  it('emits a header and one row per event', () => {
    const events = [
      makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(65) }),
      makeWorkspaceEvent({ event_id: 'b', violation_type: 'no_helmet' }),
    ];
    const csv = eventsToCsv(events);
    const lines = csv.split('\r\n');
    expect(lines[0]).toBe(EVENT_CSV_COLUMNS.join(','));
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain('a');
    expect(lines[1]).toContain('Wrong way');
  });

  it('carries the derived severity and media clock', () => {
    const row = eventCsvRow(
      makeWorkspaceEvent({ violation_type: 'no_helmet', trigger_at: mediaSeconds(65) }),
    );
    expect(row).toContain('high'); // no_helmet is high severity
    expect(row).toContain('1:05');
  });

  it('quotes cells containing commas or quotes', () => {
    const event = { ...makeWorkspaceEvent({ event_id: 'x' }), cameraId: 'cam,north "1"' };
    const csv = eventsToCsv([event]);
    expect(csv).toContain('"cam,north ""1"""');
  });
});

describe('eventsToJsonModel', () => {
  it('produces a stable, backend-shaped model', () => {
    const model = eventsToJsonModel([makeWorkspaceEvent({ event_id: 'a' })]);
    expect(model[0]).toMatchObject({
      event_id: 'a',
      violation_type: 'wrong_way',
      severity: 'medium',
      camera_id: 'cam-north',
    });
  });
});

describe('exportFilename', () => {
  it('sanitizes the base and appends the extension', () => {
    expect(exportFilename('trafficpulse events 3', 'csv')).toBe('trafficpulse-events-3.csv');
    expect(exportFilename('event-vid/../x', 'json')).toBe('event-vid-..-x.json');
    expect(exportFilename('', 'json')).toBe('trafficpulse.json');
  });
});

describe('jsonString', () => {
  it('pretty-prints JSON', () => {
    expect(jsonString({ a: 1 })).toBe('{\n  "a": 1\n}');
  });
});

describe('downloadTextFile', () => {
  const realCreate = URL.createObjectURL;
  const realRevoke = URL.revokeObjectURL;

  afterEach(() => {
    URL.createObjectURL = realCreate;
    URL.revokeObjectURL = realRevoke;
  });

  it('creates and revokes an object URL and clicks an anchor', () => {
    const createObjectURL = vi.fn(() => 'blob:x');
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = revokeObjectURL as unknown as typeof URL.revokeObjectURL;
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    const ok = downloadTextFile('events.csv', 'a,b\n1,2', 'text/csv');

    expect(ok).toBe(true);
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:x');
    clickSpy.mockRestore();
  });
});
