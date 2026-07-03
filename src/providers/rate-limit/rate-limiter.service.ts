import { randomUUID } from 'crypto';

import { Inject, Injectable, OnModuleInit } from '@nestjs/common';
import type { RedisClientType } from 'redis';

import { REDIS } from '@/providers/cache/redis';

// A held permit is one member of a per-key ZSET (leaseId -> deadline ms). Each
// acquire first reaps expired leases (crash-safe recovery), then admits iff the
// live cardinality is below the limit. Permits-in-use == ZCARD
const ACQUIRE_LUA = `
local now = tonumber(ARGV[1])
local maxConcurrent = tonumber(ARGV[2])
local leaseId = ARGV[3]
local leaseMs = tonumber(ARGV[4])
redis.call('zremrangebyscore', KEYS[1], '-inf', '(' .. now)
if redis.call('zcard', KEYS[1]) < maxConcurrent then
  redis.call('zadd', KEYS[1], now + leaseMs, leaseId)
  redis.call('pexpire', KEYS[1], leaseMs)
  return 1
end
return 0
`;

// zrem the last member auto-deletes the key, so no explicit TTL upkeep here.
const RELEASE_LUA = `
redis.call('zrem', KEYS[1], ARGV[1])
return 1
`;

// A crashed holder's permit is reclaimed within this window.
//
// No lease renewal / heartbeat: a lease is written once at acquire with a fixed
// deadline and never extended while `fn` runs. If a single gated call outlives
// LEASE_MS, its permit is reaped mid-flight, a waiter is admitted in its place,
// and the concurrency bound is transiently exceeded (the late release is then a
// harmless no-op). This holds because gated calls are individual LLM requests
// (seconds), well under 5 min. Renewal would need a background timer re-adding
// the deadline while `fn` is in flight.
// TODO(prod): when moving to managed Redis (e.g. Sentinel/failover), revisit
// this. Bottleneck (https://github.com/maselious/bottleneck) has Redis Sentinel
// support and clustered rate limiting, and is the likely path if we need it.
const LEASE_MS = 5 * 60 * 1000;

// Poll cadence for a waiter blocked on a full pool. LLM calls run for seconds,
// so ~100ms adds negligible latency
const POLL_INTERVAL_MS = 100;

/**
 * Distributed per-credential concurrency limiter backed by Redis, so the bound
 * holds across worker processes. Callers gate an arbitrary provider call via
 * {@link schedule} (e.g. embeddings). Each pool is identified by a `key`.
 */
@Injectable()
export class RateLimiterService implements OnModuleInit {
  private acquireSha = '';
  private releaseSha = '';

  constructor(@Inject(REDIS) private readonly redis: RedisClientType) {}

  async onModuleInit(): Promise<void> {
    this.acquireSha = await this.redis.scriptLoad(ACQUIRE_LUA);
    this.releaseSha = await this.redis.scriptLoad(RELEASE_LUA);
  }

  /**
   * Runs `fn` once a permit for `key` is held, releasing it in a `finally`.
   * Rejects with the signal's reason if `signal` aborts while queued.
   * `onAdmitted` receives the milliseconds spent waiting for the permit (0 when
   * admitted on the first attempt), so callers can record queue pressure
   */
  async schedule<T>(
    key: string,
    limit: number,
    signal: AbortSignal | undefined,
    fn: () => Promise<T>,
    onAdmitted?: (waitMs: number) => void,
  ): Promise<T> {
    const { leaseId, waitMs } = await this.acquire(key, limit, signal);
    onAdmitted?.(waitMs);
    try {
      return await fn();
    } finally {
      await this.release(key, leaseId);
    }
  }

  private async acquire(
    key: string,
    limit: number,
    signal: AbortSignal | undefined,
  ): Promise<{ leaseId: string; waitMs: number }> {
    const leasesKey = this.leasesKey(key);
    const leaseId = randomUUID();
    const queueStart = Date.now();
    for (;;) {
      signal?.throwIfAborted();
      const admitted = await this.runAcquire(leasesKey, limit, leaseId);
      if (admitted) return { leaseId, waitMs: Date.now() - queueStart };
      await sleep(POLL_INTERVAL_MS);
    }
  }

  private async release(key: string, leaseId: string): Promise<void> {
    await this.run(this.releaseSha, RELEASE_LUA, [this.leasesKey(key)], [leaseId]);
  }

  private async runAcquire(
    leasesKey: string,
    limit: number,
    leaseId: string,
  ): Promise<boolean> {
    // TODO(prod): `now` is each worker's wall clock, so expiry/reaping assumes
    // NTP-synced hosts; a skewed clock could reap a peer's live lease early.
    // Redis TIME inside the script would make expiry use a single clock source.
    const reply = await this.run(
      this.acquireSha,
      ACQUIRE_LUA,
      [leasesKey],
      [String(Date.now()), String(limit), leaseId, String(LEASE_MS)],
    );
    return reply === 1;
  }

  private leasesKey(key: string): string {
    return `ratelimit:sem:${key}:leases`;
  }

  // Prefer EVALSHA; self-heal on NOSCRIPT (e.g. after a Redis restart / SCRIPT
  // FLUSH) by falling back to EVAL, which re-caches the script under its sha.
  private async run(
    sha: string,
    script: string,
    keys: string[],
    args: string[],
  ): Promise<number> {
    try {
      return Number(await this.redis.evalSha(sha, { keys, arguments: args }));
    } catch (e) {
      if (e instanceof Error && e.message.includes('NOSCRIPT')) {
        return Number(await this.redis.eval(script, { keys, arguments: args }));
      }
      throw e;
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
