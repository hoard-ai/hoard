import type { Uuid } from '@/common/schemas';

import {
  buildDirectedIdMap,
  compressIdMap,
  CountingSemaphore,
  remapEdgeEndpointsToCanonical,
  UnionFind,
  withConcurrency,
} from './batch-utils';
import { RelationshipTypeSchema } from './types';

const u = (s: string) => s as Uuid;

describe('withConcurrency', () => {
  it('returns results in input order', async () => {
    const tasks = [3, 1, 2].map((n) => () => Promise.resolve(n));
    const results = await withConcurrency(2, tasks);
    expect(results).toEqual([3, 1, 2]);
  });

  it('handles an empty task array', async () => {
    const results = await withConcurrency(5, []);
    expect(results).toEqual([]);
  });

  it('handles a single task', async () => {
    const results = await withConcurrency(5, [() => Promise.resolve(42)]);
    expect(results).toEqual([42]);
  });

  it('runs no more than limit tasks concurrently', async () => {
    let active = 0;
    let maxActive = 0;
    const limit = 3;
    const tasks = Array.from({ length: 9 }, () => () => {
      active++;
      maxActive = Math.max(maxActive, active);
      return Promise.resolve().then(() => {
        active--;
        return 1;
      });
    });
    await withConcurrency(limit, tasks);
    expect(maxActive).toBeLessThanOrEqual(limit);
  });

  it('shares one concurrency budget across calls given the same semaphore', async () => {
    let active = 0;
    let maxActive = 0;
    const limit = 3;
    const semaphore = new CountingSemaphore(limit);
    const makeTasks = () =>
      Array.from({ length: 6 }, () => () => {
        active++;
        maxActive = Math.max(maxActive, active);
        return Promise.resolve().then(() => {
          active--;
          return 1;
        });
      });
    const [first, second] = await Promise.all([
      withConcurrency(semaphore, makeTasks()),
      withConcurrency(semaphore, makeTasks()),
    ]);
    expect(maxActive).toBeLessThanOrEqual(limit);
    expect(first).toHaveLength(6);
    expect(second).toHaveLength(6);
  });

  it('rethrows a task rejection', async () => {
    const err = new Error('task failed');
    const tasks = [
      () => Promise.resolve(1),
      () => Promise.reject(err),
      () => Promise.resolve(3),
    ];
    await expect(withConcurrency(5, tasks)).rejects.toThrow('task failed');
  });
});

describe('UnionFind', () => {
  it('find returns self for singleton', () => {
    const uf = new UnionFind(['a']);
    expect(uf.find('a')).toBe('a');
  });

  it('union merges two elements', () => {
    const uf = new UnionFind(['a', 'b']);
    uf.union('a', 'b');
    expect(uf.find('a')).toBe(uf.find('b'));
  });

  it('lex-smaller id wins as root', () => {
    const uf = new UnionFind(['b', 'a']);
    uf.union('b', 'a');
    expect(uf.find('b')).toBe('a');
    expect(uf.find('a')).toBe('a');
  });

  it('path compression flattens long chain', () => {
    const uf = new UnionFind(['a', 'b', 'c', 'd']);
    uf.union('b', 'a'); // b → a
    uf.union('c', 'b'); // c → a (via b)
    uf.union('d', 'c'); // d → a (via c)
    expect(uf.find('d')).toBe('a');
    // After path compression, find should still return same root
    expect(uf.find('c')).toBe('a');
    expect(uf.find('b')).toBe('a');
  });

  it('idempotent union has no effect', () => {
    const uf = new UnionFind(['a', 'b']);
    uf.union('a', 'b');
    uf.union('a', 'b');
    expect(uf.find('a')).toBe(uf.find('b'));
  });
});

describe('compressIdMap', () => {
  it('single pair: smaller id is canonical', () => {
    const map = compressIdMap([['b', 'a']]);
    expect(map.get('a')).toBe('a');
    expect(map.get('b')).toBe('a');
  });

  it('chain a→b→c collapses to lex-min', () => {
    // All three end up in the same cluster; lex-min is 'a'
    const map = compressIdMap([
      ['b', 'a'],
      ['c', 'b'],
    ]);
    expect(map.get('a')).toBe('a');
    expect(map.get('b')).toBe('a');
    expect(map.get('c')).toBe('a');
  });

  it('disjoint pairs stay separate', () => {
    const map = compressIdMap([
      ['b', 'a'],
      ['d', 'c'],
    ]);
    expect(map.get('a')).toBe('a');
    expect(map.get('b')).toBe('a');
    expect(map.get('c')).toBe('c');
    expect(map.get('d')).toBe('c');
  });
});

describe('buildDirectedIdMap', () => {
  it('directed pair b→a maps b to a', () => {
    const map = buildDirectedIdMap([['b', 'a']]);
    expect(map.get('b')).toBe('a');
    expect(map.get('a')).toBe('a');
  });

  it('chain b→a, c→b collapses to a', () => {
    const map = buildDirectedIdMap([
      ['b', 'a'],
      ['c', 'b'],
    ]);
    expect(map.get('c')).toBe('a');
    expect(map.get('b')).toBe('a');
  });

  it('independent pairs stay separate', () => {
    const map = buildDirectedIdMap([
      ['b', 'a'],
      ['d', 'c'],
    ]);
    expect(map.get('b')).toBe('a');
    expect(map.get('d')).toBe('c');
  });
});

const makeEdge = (id: string, sourceNodeId: string, targetNodeId: string) => ({
  id: u(id),
  sourceNodeId: u(sourceNodeId),
  targetNodeId: u(targetNodeId),
  name: RelationshipTypeSchema.parse('RELATES_TO'),
  fact: 'test',
  graphId: 'g' as Uuid,
  episodes: [],
  createdAt: new Date(),
  validAt: null,
  invalidAt: null,
  expiredAt: null,
  factEmbedding: null,
  attributes: {},
});

describe('remapEdgeEndpointsToCanonical', () => {
  it('remaps sourceNodeId and targetNodeId via map', () => {
    const edge = makeEdge('e1', 'old-src', 'old-tgt');
    const idMap = new Map<Uuid, Uuid>([
      [u('old-src'), u('new-src')],
      [u('old-tgt'), u('new-tgt')],
    ]);
    const [result] = remapEdgeEndpointsToCanonical([edge], idMap);
    expect(result.sourceNodeId).toBe('new-src');
    expect(result.targetNodeId).toBe('new-tgt');
  });

  it('unmapped ids are unchanged', () => {
    const edge = makeEdge('e1', 'src', 'tgt');
    const idMap = new Map<Uuid, Uuid>();
    const [result] = remapEdgeEndpointsToCanonical([edge], idMap);
    expect(result.sourceNodeId).toBe('src');
    expect(result.targetNodeId).toBe('tgt');
  });

  it('does not mutate input edge', () => {
    const edge = makeEdge('e1', 'old-src', 'old-tgt');
    const idMap = new Map<Uuid, Uuid>([[u('old-src'), u('new-src')]]);
    remapEdgeEndpointsToCanonical([edge], idMap);
    expect(edge.sourceNodeId).toBe('old-src');
  });
});
