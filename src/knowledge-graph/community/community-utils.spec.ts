import type { Uuid } from '@/common/schemas';
import { u } from '@/test/factories';

import {
  type ClusterMatch,
  type ExistingCommunitySnapshot,
} from '../repository/repositories';
import type { NodeName } from '../types';
import { findUniqueSuperset, planRoutes, summaryHash } from './community-utils';

function snap(overrides: Partial<ExistingCommunitySnapshot>): ExistingCommunitySnapshot {
  return {
    id: u('default-existing'),
    name: 'existing-community' as NodeName,
    memberIds: [],
    summary: '',
    nameEmbedding: null,
    ...overrides,
  };
}

function hashesFor(
  memberIds: Uuid[],
  summaryById: Map<Uuid, string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const id of memberIds) out[id] = summaryHash(summaryById.get(id) ?? '');
  return out;
}

function sameMembers(a: Uuid[], b: Uuid[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((x, i) => x === sb[i]);
}

describe('findUniqueSuperset', () => {
  const cluster = [u('a'), u('b'), u('c')];

  it('returns the existing community when its members are a strict subset of the cluster', () => {
    const existing = snap({ id: u('existing-1'), memberIds: [u('a'), u('b')] });
    expect(findUniqueSuperset(cluster, [existing])).toBe(existing);
  });

  it('returns null when no existing has overlap with the cluster', () => {
    const unrelated = snap({ id: u('unrelated'), memberIds: [u('x'), u('y')] });
    expect(findUniqueSuperset(cluster, [unrelated])).toBeNull();
  });

  it('returns null on set-equality (matched-set already handled that case)', () => {
    const equal = snap({ id: u('equal'), memberIds: [...cluster] });
    expect(findUniqueSuperset(cluster, [equal])).toBeNull();
  });

  it('returns null when two existing communities both overlap the cluster (ambiguous)', () => {
    const e1 = snap({ id: u('e1'), memberIds: [u('a')] });
    const e2 = snap({ id: u('e2'), memberIds: [u('b')] });
    expect(findUniqueSuperset(cluster, [e1, e2])).toBeNull();
  });

  it('returns null when the candidate has a member not in the cluster (would be a removal)', () => {
    const withExtra = snap({
      id: u('with-extra'),
      memberIds: [u('a'), u('removed')],
    });
    expect(findUniqueSuperset(cluster, [withExtra])).toBeNull();
  });
});

describe('planRoutes', () => {
  // Two disconnected triangles. summaryById and storedHashes are constructed
  // so "stored = live" by default; tests perturb individual entries to
  // simulate drift.
  const cluster1: Uuid[] = [u('c1-a'), u('c1-b'), u('c1-c')];
  const cluster2: Uuid[] = [u('c2-a'), u('c2-b'), u('c2-c')];
  const summaryById = new Map<Uuid, string>([
    [cluster1[0], 'alice'],
    [cluster1[1], 'bob'],
    [cluster1[2], 'charlie'],
    [cluster2[0], 'dave'],
    [cluster2[1], 'eve'],
    [cluster2[2], 'frank'],
  ]);

  function matchSet(communityId: Uuid, members: Uuid[]): ClusterMatch {
    return {
      kind: 'matched-set',
      communityId,
      storedHashes: hashesFor(members, summaryById),
    };
  }

  it('produces all-clean routes when every matched-set cluster has stored hashes equal to fresh', () => {
    const existing1 = u('existing-1');
    const existing2 = u('existing-2');
    const matches = new Map<number, ClusterMatch>([
      [0, matchSet(existing1, cluster1)],
      [1, matchSet(existing2, cluster2)],
    ]);
    const existing = [
      snap({ id: existing1, memberIds: cluster1, summary: 's1' }),
      snap({ id: existing2, memberIds: cluster2, summary: 's2' }),
    ];

    const routes = planRoutes([cluster1, cluster2], matches, existing, summaryById);

    expect(routes).toHaveLength(2);
    expect(routes.every((r) => r.kind === 'clean')).toBe(true);
  });

  it('emits an incremental route with only the drifted member summaries when a matched set has stale hashes', () => {
    const existing1 = u('existing-1');
    const driftedSummaries = new Map(summaryById);
    // Stored hash for c1-b is stale (it reflects an older summary).
    const staleHashes = hashesFor(cluster1, summaryById);
    staleHashes[cluster1[1]] = 'STALE_HASH_FOR_BOB';

    const matches = new Map<number, ClusterMatch>([
      [
        0,
        {
          kind: 'matched-set',
          communityId: existing1,
          storedHashes: staleHashes,
        },
      ],
    ]);
    const existing = [
      snap({ id: existing1, memberIds: cluster1, summary: 'old-summary' }),
    ];

    const routes = planRoutes([cluster1], matches, existing, driftedSummaries);

    expect(routes).toHaveLength(1);
    const r = routes[0];
    if (r.kind !== 'incremental') throw new Error(`expected incremental, got ${r.kind}`);
    expect(r.communityId).toBe(existing1);
    expect(r.existingSummary).toBe('old-summary');
    expect(sameMembers(r.finalMemberIds, cluster1)).toBe(true);
    // Only c1-b ('bob') drifted - delta should be exactly its fresh summary.
    expect(r.deltaSummaries).toEqual(['bob']);
  });

  it('emits an incremental route on a pure-addition superset, with only the added members as deltas', () => {
    const existingId = u('existing-superset');
    const partialMembers = [cluster2[0], cluster2[1]];
    const matches = new Map<number, ClusterMatch>([[0, { kind: 'unmatched' }]]);
    const existing = [
      snap({ id: existingId, memberIds: partialMembers, summary: 'old-superset' }),
    ];

    const routes = planRoutes([cluster2], matches, existing, summaryById);

    expect(routes).toHaveLength(1);
    const r = routes[0];
    if (r.kind !== 'incremental') throw new Error(`expected incremental, got ${r.kind}`);
    expect(r.communityId).toBe(existingId);
    expect(r.existingSummary).toBe('old-superset');
    expect(sameMembers(r.finalMemberIds, cluster2)).toBe(true);
    // Only the third (added) member's summary feeds the delta tournament.
    expect(r.deltaSummaries).toEqual([summaryById.get(cluster2[2])]);
  });

  it('falls back to a full route when an existing community has members not in the cluster (removal)', () => {
    const existingWithExtra = u('existing-with-removed');
    const removed = u('removed-member');
    const matches = new Map<number, ClusterMatch>([[0, { kind: 'unmatched' }]]);
    const existing = [
      snap({
        id: existingWithExtra,
        memberIds: [...cluster2, removed],
        summary: 'old',
      }),
    ];

    const routes = planRoutes([cluster2], matches, existing, summaryById);

    expect(routes).toHaveLength(1);
    const r = routes[0];
    if (r.kind !== 'full') throw new Error(`expected full, got ${r.kind}`);
    expect(sameMembers(r.memberIds, cluster2)).toBe(true);
    expect(r.memberSummaries.sort()).toEqual(
      cluster2.map((id) => summaryById.get(id)!).sort(),
    );
  });

  it('falls back to a full route when a cluster supersets multiple existing communities (ambiguous merge)', () => {
    const merged: Uuid[] = [...cluster1, ...cluster2].sort();
    const e1 = u('e1');
    const e2 = u('e2');
    const matches = new Map<number, ClusterMatch>([[0, { kind: 'unmatched' }]]);
    const existing = [
      snap({ id: e1, memberIds: cluster1, summary: 's1' }),
      snap({ id: e2, memberIds: cluster2, summary: 's2' }),
    ];

    const routes = planRoutes([merged], matches, existing, summaryById);

    expect(routes).toHaveLength(1);
    const r = routes[0];
    if (r.kind !== 'full') throw new Error(`expected full, got ${r.kind}`);
    expect(sameMembers(r.memberIds, merged)).toBe(true);
  });

  it('reserves an existing community for one cluster so a later cluster cannot also claim it as its superset', () => {
    // Two clusters both supersetting the same existing community would be
    // illegal because membership uniquely identifies a community. Whichever
    // cluster comes first wins; the other falls back to full.
    const sharedA = u('shared-a');
    const sharedB = u('shared-b');
    const aExtra = u('a-extra');
    const bExtra = u('b-extra');
    const sharedSubset = [sharedA, sharedB];
    const clusterA = [sharedA, sharedB, aExtra];
    const clusterB = [sharedA, sharedB, bExtra];
    const localSummaries = new Map<Uuid, string>([
      [sharedA, 's-a'],
      [sharedB, 's-b'],
      [aExtra, 's-a-extra'],
      [bExtra, 's-b-extra'],
    ]);

    const matches = new Map<number, ClusterMatch>([
      [0, { kind: 'unmatched' }],
      [1, { kind: 'unmatched' }],
    ]);
    const existing = [
      snap({ id: u('shared-existing'), memberIds: sharedSubset, summary: 'shared' }),
    ];

    const routes = planRoutes([clusterA, clusterB], matches, existing, localSummaries);

    // First cluster gets the superset incremental; second falls back to full
    // because the only candidate is reserved.
    expect(routes[0].kind).toBe('incremental');
    expect(routes[1].kind).toBe('full');
  });
});
