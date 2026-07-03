import type { RedisClientType } from 'redis';

import { RateLimiterService } from './rate-limiter.service';

// In-memory stand-in for the two Lua scripts: a per-key Set models the leases
// ZSET. Covers the acquire/release capacity check and thus the service's
// scheduling logic. Crash-lease reaping (zremrangebyscore on deadlines) is Lua
// tested against a real Redis, not here.
function createFakeRedis() {
  const leases = new Map<string, Set<string>>();
  const evalScript = (sha: string, keys: string[], args: string[]): number => {
    const key = keys[0];
    const set = leases.get(key) ?? new Set<string>();
    leases.set(key, set);
    if (sha === 'acquire') {
      const [, maxConcurrent, leaseId] = args;
      if (set.size < Number(maxConcurrent)) {
        set.add(leaseId);
        return 1;
      }
      return 0;
    }
    set.delete(args[0]);
    return 1;
  };
  const shaFor = (script: string) => (script.includes('zcard') ? 'acquire' : 'release');
  return {
    leases,
    scriptLoad: jest.fn((script: string) => Promise.resolve(shaFor(script))),
    evalSha: jest.fn((sha: string, o: { keys: string[]; arguments: string[] }) =>
      Promise.resolve(evalScript(sha, o.keys, o.arguments)),
    ),
    eval: jest.fn((script: string, o: { keys: string[]; arguments: string[] }) =>
      Promise.resolve(evalScript(shaFor(script), o.keys, o.arguments)),
    ),
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('RateLimiterService', () => {
  let service: RateLimiterService;
  let redis: ReturnType<typeof createFakeRedis>;

  beforeEach(async () => {
    redis = createFakeRedis();
    service = new RateLimiterService(redis as unknown as RedisClientType);
    await service.onModuleInit();
  });

  describe('schedule', () => {
    it('bounds concurrent execution to the limit', async () => {
      let live = 0;
      let maxLive = 0;
      const task = async () => {
        live++;
        maxLive = Math.max(maxLive, live);
        await sleep(20);
        live--;
      };

      await Promise.all(
        Array.from({ length: 8 }, () => service.schedule('k', 3, undefined, task)),
      );

      expect(maxLive).toBe(3);
    });

    it('releases the permit so later work can acquire (no leak)', async () => {
      const run = () => service.schedule('k', 1, undefined, () => sleep(5));
      await run();
      await run();
      await run();
      // With a leaked permit the third call would deadlock; completing proves release.
      expect(redis.leases.get('ratelimit:sem:k:leases')?.size ?? 0).toBe(0);
    });

    it('rejects immediately when the signal is already aborted', async () => {
      const controller = new AbortController();
      controller.abort();
      const fn = jest.fn(() => Promise.resolve('ok'));

      await expect(service.schedule('k', 1, controller.signal, fn)).rejects.toThrow();
      expect(fn).not.toHaveBeenCalled();
    });

    it('rejects queued work when the signal aborts, without running it', async () => {
      const controller = new AbortController();
      const started: number[] = [];
      const mk = (i: number) =>
        service.schedule('k', 1, controller.signal, async () => {
          started.push(i);
          await sleep(300);
        });

      const settled = Promise.allSettled([mk(0), mk(1), mk(2)]);
      await sleep(60); // let task 0 acquire and start
      controller.abort();
      const results = await settled;

      expect(results[0].status).toBe('fulfilled');
      expect(results[1].status).toBe('rejected');
      expect(results[2].status).toBe('rejected');
      expect(started).toEqual([0]); // queued tasks never entered the fn
    });
  });
});
