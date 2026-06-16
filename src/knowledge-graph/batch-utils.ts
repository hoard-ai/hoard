import { Uuid } from '@/common/schemas';
import { EntityEdge } from '@/knowledge-graph/models';

/**
 * Maximum number of concurrent LLM API calls in bulk ingestion.
 * Prevents rate-limit exhaustion when processing large episode batches.
 */
export const LLM_CONCURRENCY_LIMIT = 10;

// TODO: global LLM semaphore. withConcurrency creates a fresh semaphore per
// call, so nested fan-out (per-episode in episode.service wrapping per-chunk in
// node/edge extraction) can run up to LLM_CONCURRENCY_LIMIT^2 calls at once and
// blow past provider rate limits. Replace the per-call limit with one shared
// permit pool so total in-flight LLM calls are bounded regardless of nesting.

class CountingSemaphore {
  private count: number;
  private readonly waiters: Array<() => void> = [];

  constructor(count: number) {
    if (count < 1) throw new Error('Semaphore count must be at least 1');
    this.count = count;
  }

  async acquire(): Promise<void> {
    if (this.count > 0) {
      this.count--;
      return;
    }
    return new Promise<void>((resolve) => {
      this.waiters.push(resolve);
    });
  }

  release(): void {
    if (this.waiters.length > 0) {
      const nextResolve = this.waiters.shift();
      if (nextResolve) nextResolve();
    } else {
      this.count++;
    }
  }
}

/**
 * Runs tasks with bounded parallelism using a counting semaphore.
 *
 * All tasks are launched concurrently; the semaphore limits how many run at
 * once. Each slot is always released in a `finally` block so a rejection never
 * leaks a held permit.
 *
 * @param tasks   Zero-argument factory functions returning Promises.
 * @param limit   Maximum tasks to run concurrently.
 * @returns       Results in the same order as `tasks`.
 */
export async function withConcurrency<T>(
  limit: number,
  tasks: (() => Promise<T>)[],
): Promise<T[]> {
  const semaphore = new CountingSemaphore(limit);
  return Promise.all(
    tasks.map(async (task) => {
      await semaphore.acquire();
      try {
        return await task();
      } finally {
        semaphore.release();
      }
    }),
  );
}

export function reassembleByOffsets<T>(flat: T[], lengths: number[]): T[][] {
  let offset = 0;
  return lengths.map((len) => {
    const slice = flat.slice(offset, offset + len);
    offset += len;
    return slice;
  });
}

/**
 * Union-Find (disjoint set) with iterative path compression.
 * When merging two roots, the lexicographically larger root is attached under
 * the smaller one — so `find()` always returns the lex-smallest representative.
 */
export class UnionFind<T extends string = string> {
  private readonly parent: Map<T, T>;

  constructor(elements: Iterable<T>) {
    this.parent = new Map([...elements].map((e) => [e, e]));
  }

  find(x: T): T {
    // Iterative path compression
    let root = x;
    while (this.parent.get(root) !== root) {
      root = this.parent.get(root)!;
    }
    let current = x;
    while (current !== root) {
      const next = this.parent.get(current)!;
      this.parent.set(current, root);
      current = next;
    }
    return root;
  }

  union(a: T, b: T): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;
    // Attach lex-larger under lex-smaller
    if (ra < rb) {
      this.parent.set(rb, ra);
    } else {
      this.parent.set(ra, rb);
    }
  }
}

/**
 * Maps each ID to the lexicographically smallest canonical ID in its
 * duplicate cluster. Used for within-batch bidirectional deduplication.
 */
export function compressIdMap<T extends string = string>(pairs: [T, T][]): Map<T, T> {
  const allIds = new Set<T>();
  for (const [a, b] of pairs) {
    allIds.add(a);
    allIds.add(b);
  }

  const uf = new UnionFind<T>(allIds);
  for (const [a, b] of pairs) {
    uf.union(a, b);
  }
  return new Map([...allIds].map((id) => [id, uf.find(id)]));
}

/**
 * Builds a directed alias → canonical map from (extractedId, canonicalId)
 * pairs returned by node resolution. Uses union-find with path compression so
 * chains of aliases collapse to their ultimate canonical target.
 */
export function buildDirectedIdMap<T extends string = string>(
  pairs: [T, T][],
): Map<T, T> {
  const parent = new Map<T, T>();

  const find = (id: T): T => {
    if (!parent.has(id)) parent.set(id, id);
    let root = id;
    while (parent.get(root) !== root) root = parent.get(root)!;
    let cur = id;
    while (cur !== root) {
      const next = parent.get(cur)!;
      parent.set(cur, root);
      cur = next;
    }
    return root;
  };

  for (const [src, tgt] of pairs) {
    if (!parent.has(src)) parent.set(src, src);
    if (!parent.has(tgt)) parent.set(tgt, tgt);
    parent.set(find(src), find(tgt));
  }
  return new Map([...parent.keys()].map((id) => [id, find(id)]));
}

/**
 * Remaps edge source/target node IDs through a deduplication map.
 * Returns new edge objects; does not mutate input.
 */
export function resolveEdgePointers(
  edges: EntityEdge[],
  idMap: Map<Uuid, Uuid>,
): EntityEdge[] {
  return edges.map((e) => ({
    ...e,
    sourceNodeId: idMap.get(e.sourceNodeId) ?? e.sourceNodeId,
    targetNodeId: idMap.get(e.targetNodeId) ?? e.targetNodeId,
  }));
}
