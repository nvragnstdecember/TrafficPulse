import { describe, expect, it } from 'vitest';

import { type LiveMotorcycle, type LiveRider } from '@/api/types';
import { makeConfirmedEvent, mediaSeconds } from '@/test/fixtures';

import {
  cadenceSentence,
  cameraErrorMessage,
  formatLatency,
  formatRate,
  formatUptime,
  helmetLabel,
  helmetTone,
  liveStatus,
  liveStatusTone,
  occupancyLabel,
  occupancyTone,
  parseServerMessage,
  socketUrl,
  toEventRow,
} from './live';

function motorcycle(overrides: Partial<LiveMotorcycle> = {}): LiveMotorcycle {
  return {
    motorcycle_track_id: 'trk-1',
    rider_count: 1,
    driver_resolved: true,
    ...overrides,
  };
}

function rider(overrides: Partial<LiveRider> = {}): LiveRider {
  return {
    rider_track_id: 'trk-2',
    motorcycle_track_id: 'trk-1',
    rider_count: 1,
    driver_resolved: true,
    helmet_label: 'helmet',
    helmet_confidence: 0.9,
    helmet_gated: false,
    ...overrides,
  };
}

describe('live status', () => {
  it('keeps the camera and the session as separate lifecycles', () => {
    // A camera can be previewing with no session, which is the state the whole
    // "monitoring is an explicit act" design depends on being expressible.
    expect(liveStatus('ready', 'idle')).toBe('camera ready');
    expect(liveStatus('off', 'idle')).toBe('camera off');
    expect(liveStatus('ready', 'monitoring')).toBe('monitoring');
    expect(liveStatus('ready', 'connecting')).toBe('connecting');
  });

  it('reports a session left without a camera as disconnected, not idle', () => {
    expect(liveStatus('off', 'monitoring')).toBe('disconnected');
    expect(liveStatusTone('disconnected')).toBe('warning');
  });

  it('surfaces an error from either lifecycle', () => {
    expect(liveStatus('error', 'idle')).toBe('error');
    expect(liveStatus('ready', 'error')).toBe('error');
    expect(liveStatusTone('error')).toBe('error');
    expect(liveStatusTone('monitoring')).toBe('success');
  });
});

describe('camera errors', () => {
  it('turns each browser failure into the recovery that actually applies', () => {
    const denied = cameraErrorMessage(
      Object.assign(new Error('Permission denied'), { name: 'NotAllowedError' }),
    );
    expect(denied).toMatch(/allow camera access/i);

    const missing = cameraErrorMessage(
      Object.assign(new Error('no device'), { name: 'NotFoundError' }),
    );
    expect(missing).toMatch(/no camera was found/i);

    const busy = cameraErrorMessage(
      Object.assign(new Error('busy'), { name: 'NotReadableError' }),
    );
    expect(busy).toMatch(/another application/i);
  });

  it('falls back to the browser message rather than an empty string', () => {
    expect(cameraErrorMessage(new Error('something odd'))).toContain('something odd');
    expect(cameraErrorMessage('nonsense')).toBe('The camera could not be started.');
  });
});

describe('socketUrl', () => {
  it('derives ws:// and wss:// from the page origin', () => {
    expect(socketUrl('/api/live/ws', '', 'http://localhost:5173')).toBe(
      'ws://localhost:5173/api/live/ws',
    );
    expect(socketUrl('/api/live/ws', '', 'https://traffic.example')).toBe(
      'wss://traffic.example/api/live/ws',
    );
  });

  it('honours a configured API base URL over the page origin', () => {
    expect(socketUrl('/api/live/ws', 'https://api.example', 'http://localhost:5173')).toBe(
      'wss://api.example/api/live/ws',
    );
  });
});

describe('parseServerMessage', () => {
  it('accepts every message the protocol defines', () => {
    for (const type of ['session', 'result', 'events', 'warning', 'error', 'stopped']) {
      expect(parseServerMessage({ type })?.type).toBe(type);
    }
  });

  it('ignores an unknown message instead of failing the session', () => {
    // A newer server is a reason to skip a message, not to drop the connection.
    expect(parseServerMessage({ type: 'something-new' })).toBeNull();
    expect(parseServerMessage(null)).toBeNull();
    expect(parseServerMessage('result')).toBeNull();
  });
});

describe('toEventRow', () => {
  it('carries the observation window the reasoner sustained', () => {
    const received = new Date('2026-01-01T10:00:00Z');
    const row = toEventRow(
      makeConfirmedEvent({
        event_id: 'evt-9',
        violation_type: 'triple_riding',
        track_ids: ['t-4'],
        start_at: mediaSeconds(8),
        trigger_at: mediaSeconds(10.5),
      }),
      received,
    );
    expect(row.id).toBe('evt-9');
    expect(row.label).toBe('Triple riding');
    expect(row.observedSeconds).toBeCloseTo(2.5);
    expect(row.receivedAt).toBe(received);
  });
});

describe('occupancy and helmet readings', () => {
  it('states multi-rider driver attribution as unresolved, in words', () => {
    // The whole point: a viewer must be told the driver cannot be identified,
    // rather than inferring it from a missing label.
    const many = motorcycle({ rider_count: 3, driver_resolved: false });
    expect(occupancyLabel(many)).toBe('3 riders — DRIVER UNRESOLVED');
    expect(occupancyTone(many)).toBe('warning');

    expect(occupancyLabel(motorcycle())).toBe('1 rider');
    expect(occupancyTone(motorcycle())).toBe('neutral');
  });

  it('distinguishes a refused crop from an absent classifier and from a reading', () => {
    expect(helmetLabel(rider({ helmet_gated: true }))).toBe('crop refused');
    expect(helmetLabel(rider({ helmet_label: null }))).toBe('not classified');
    expect(helmetLabel(rider({ helmet_label: 'no_helmet' }))).toBe('no helmet');

    // A refused crop is neutral, not an alarm: nothing was read.
    expect(helmetTone(rider({ helmet_gated: true }))).toBe('neutral');
    expect(helmetTone(rider({ helmet_label: 'no_helmet' }))).toBe('warning');
    expect(helmetTone(rider({ helmet_label: 'helmet' }))).toBe('success');
  });
});

describe('measured readings', () => {
  it('renders an unmeasured value as an em dash, never as zero', () => {
    expect(formatRate(null)).toBe('—');
    expect(formatRate(undefined)).toBe('—');
    expect(formatLatency(null)).toBe('—');
    expect(formatRate(1.83)).toBe('1.8 fps');
    expect(formatLatency(1740)).toBe('1.74 s');
    expect(formatLatency(320)).toBe('320 ms');
    expect(formatUptime(125)).toBe('2:05');
  });

  it('says in words that the camera rate is not the inference rate', () => {
    const sentence = cadenceSentence(30, 1.6);
    expect(sentence).toContain('30 fps');
    expect(sentence).toContain('1.6');
    expect(sentence).toMatch(/dropped, never queued/);
  });

  it('claims no rate before one has been measured', () => {
    expect(cadenceSentence(30, null)).toMatch(/has not processed a frame yet/);
  });
});
