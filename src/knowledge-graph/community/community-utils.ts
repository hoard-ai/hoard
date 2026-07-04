import { createHash } from 'node:crypto';

import type { Uuid } from '@/common/schemas';

import type { ClusterMatch, ExistingCommunitySnapshot } from '../repository/repositories';

/**
 * Mulberry32 seeded PRNG (returns a function yielding floats in [0, 1)). Feeding
 * a seeded rng to Louvain makes community detection deterministic: the same
 * graph yields the same partition every run, so member signatures stay stable
 * and rebuilds don't churn communities just because of traversal randomness.
 */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * sha256-hex of a single entity_nodes.summary value. MUST match the formula in
 * `compute_community_member_summary_hashes` (migration SQL) so stored snapshots
 * are comparable to TS-computed fresh hashes.
 */
export function summaryHash(summary: string): string {
  return createHash('sha256').update(summary, 'utf8').digest('hex');
}

/**
 * Canonical form for community-name equality checks.
 */
export function normalizeName(name: string): string {
  return name.trim().toLowerCase();
}

/**
 * Among `existing` communities, find the single one whose memberIds is a strict
 * subset of `clusterMemberIds` AND such that no other existing community has
 * any overlap with the cluster. Returns null if no such unique superset exists.
 *
 * The "no overlap with any other" guard rejects ambiguous merges: a cluster
 * that supersets one community while sharing members with another can't be
 * safely seeded from a single existing summary.
 */
export function findUniqueSuperset<T extends { id: Uuid; memberIds: Uuid[] }>(
  clusterMemberIds: Uuid[],
  existing: T[],
): T | null {
  const clusterSet = new Set(clusterMemberIds);

  const overlappingExisting = existing.filter((e) =>
    e.memberIds.some((id) => clusterSet.has(id)),
  );
  if (overlappingExisting.length !== 1) return null;

  const candidate = overlappingExisting[0];
  // Strict subset: every existing member is in cluster, and cluster has at
  // least one member outside existing (otherwise it's set-equality, which
  // matched-set already handled).
  const allIn = candidate.memberIds.every((id) => clusterSet.has(id));
  if (!allIn) return null;
  if (candidate.memberIds.length >= clusterMemberIds.length) return null;
  return candidate;
}

/**
 * What to do for a single detected cluster during buildCommunities:
 *
 * - `clean`: matched an existing community by set + all member-summary hashes
 *   agree with the stored snapshot. No LLM, no write.
 * - `incremental`: either (a) same-set match with some drifted hashes, or
 *   (b) cluster strictly supersets a unique existing community. Reuses the
 *   existing community summary as the tournament seed and feeds the
 *   drifted-or-added members' fresh summaries as deltas, then updates in place.
 * - `full`: no usable existing seed (no match, removal, or ambiguous superset).
 *   Runs the full hierarchical tournament and creates a new community row.
 * - `skip`: cluster has no usable input (all member summaries empty, typically
 *   entities not yet summarized). No LLM, no write. Re-considered next rebuild.
 */
export type ClusterRoute =
  | { kind: 'clean'; communityId: Uuid }
  | {
      kind: 'incremental';
      communityId: Uuid;
      existingSummary: string;
      finalMemberIds: Uuid[];
      deltaSummaries: string[];
    }
  | { kind: 'full'; memberIds: Uuid[]; memberSummaries: string[] }
  | { kind: 'skip'; reason: 'no-member-summaries' };

/**
 * Pure routing: given detected clusters + match results + existing community
 * snapshots + fresh per-member summaries, decide what to do for each cluster.
 *
 * The reservation set is internal: clusters are processed in order so a later
 * cluster cannot claim an existing community already consumed by an earlier
 * matched-set or superset route. The caller can derive the final consumed set
 * from `routes.filter(r => r.kind !== 'full').map(r => r.communityId)`.
 */
export function planRoutes(
  clusters: Uuid[][],
  matchesByClusterIndex: Map<number, ClusterMatch>,
  existing: ExistingCommunitySnapshot[],
  summaryById: Map<Uuid, string>,
): ClusterRoute[] {
  const lookupSummary = (id: Uuid): string => summaryById.get(id) ?? '';
  const existingById = new Map(existing.map((e) => [e.id, e]));
  const consumedExistingIds = new Set<Uuid>();

  return clusters.map((memberIds, idx): ClusterRoute => {
    const match = matchesByClusterIndex.get(idx);

    if (match?.kind === 'matched-set') {
      consumedExistingIds.add(match.communityId);
      const driftedIds = memberIds.filter(
        (id) => summaryHash(lookupSummary(id)) !== match.storedHashes[id],
      );
      if (driftedIds.length === 0) {
        return { kind: 'clean', communityId: match.communityId };
      }
      const snap = existingById.get(match.communityId);
      return {
        kind: 'incremental',
        communityId: match.communityId,
        existingSummary: snap?.summary ?? '',
        finalMemberIds: memberIds,
        deltaSummaries: driftedIds.map(lookupSummary).filter((s) => s.length > 0),
      };
    }

    // Unmatched-by-set: try the addition-only path. consumedExistingIds
    // doubles as the reservation set so two clusters can't claim the same
    // existing community as their superset seed.
    const superset = findUniqueSuperset(
      memberIds,
      existing.filter((e) => !consumedExistingIds.has(e.id)),
    );
    if (superset) {
      consumedExistingIds.add(superset.id);
      const existingMemberSet = new Set(superset.memberIds);
      const addedIds = memberIds.filter((id) => !existingMemberSet.has(id));
      return {
        kind: 'incremental',
        communityId: superset.id,
        existingSummary: superset.summary,
        finalMemberIds: memberIds,
        deltaSummaries: addedIds.map(lookupSummary).filter((s) => s.length > 0),
      };
    }

    const memberSummaries = memberIds.map(lookupSummary).filter((s) => s.length > 0);
    if (memberSummaries.length === 0) {
      return { kind: 'skip', reason: 'no-member-summaries' };
    }
    return { kind: 'full', memberIds, memberSummaries };
  });
}
