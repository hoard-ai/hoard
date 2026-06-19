import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';
import { UndirectedGraph } from 'graphology';
import louvain from 'graphology-communities-louvain';

import type { Uuid } from '@/common/schemas';
import { invokeStructured } from '@/llm';
import { LlmService } from '@/llm/llm.service';
import {
  LLM_TRACER,
  type LlmContext,
  type LlmTracer,
  metricsOnResult,
  Span,
  type SpanMetrics,
} from '@/observability';

import { LLM_CONCURRENCY_LIMIT, withConcurrency } from '../batch-utils';
import { EmbeddingService } from '../embedding';
import { Community, createCommunity } from '../models';
import {
  buildCommunityNameMessages,
  buildResolveNameCollisionsMessages,
  buildResolveNameCollisionsValidator,
  buildSummarizePairMessages,
  type Collider,
  CommunityNameSchema,
  ResolveNameCollisionsSchema,
  SummarySchema,
} from '../prompts';
import { MAX_SUMMARY_CHARS, truncateAtSentence } from '../prompts/text-utils';
import {
  CommunityRepository,
  EntityEdgeRepository,
  EntityNodeRepository,
} from '../repository/repositories';
import { NodeNameSchema } from '../types';
import {
  type ClusterRoute,
  mulberry32,
  normalizeName,
  planRoutes,
} from './community-utils';

/**
 * Fixed seed for Louvain's random walk.
 */
const LOUVAIN_SEED = 42;

/**
 * Per-route pending result of the parallel reduce-and-name phase. Embedding
 * and persistence happen after collision resolution in a single post-pass.
 */
type PendingRoute =
  | { kind: 'noop' }
  | {
      kind: 'full';
      tempId: number;
      name: string;
      summary: string;
      memberIds: Uuid[];
    }
  | {
      kind: 'incremental';
      tempId: number;
      communityId: Uuid;
      name: string;
      summary: string;
      finalMemberIds: Uuid[];
    };

@Injectable()
export class CommunityService {
  constructor(
    private readonly llmService: LlmService,
    private readonly embeddingService: EmbeddingService,
    private readonly communityRepo: CommunityRepository,
    private readonly entityNodeRepo: EntityNodeRepository,
    private readonly entityEdgeRepo: EntityEdgeRepository,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  async buildCommunities(userId: Uuid, graphId: Uuid, ctx?: LlmContext): Promise<void> {
    await this.buildCommunitiesImpl(userId, graphId, ctx);
  }

  @Span('buildCommunities', { onResult: metricsOnResult })
  private async buildCommunitiesImpl(
    userId: Uuid,
    graphId: Uuid,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const startMs = performance.now();
    const baseMetrics: SpanMetrics = {
      'user.id': userId,
      'session.id': ctx?.sessionId,
      'graph.id': graphId,
    };
    const model = await this.llmService.getActiveModel(userId);

    // 1. Build the weighted graph and detect communities via Louvain.
    const nodeIds = await this.entityNodeRepo.findIdsForGraph(graphId);

    if (nodeIds.length < 2) {
      await this.communityRepo.deleteByGraphId(graphId);
      return {
        metrics: {
          ...baseMetrics,
          'nodes.count': nodeIds.length,
          skipped: true,
          'skipped.reason': 'too-few-nodes',
          duration_ms: Math.round(performance.now() - startMs),
        },
      };
    }

    const aggregatedEdges =
      await this.entityEdgeRepo.findAggregatedNeighborCounts(graphId);

    if (aggregatedEdges.length === 0) {
      await this.communityRepo.deleteByGraphId(graphId);
      return {
        metrics: {
          ...baseMetrics,
          'nodes.count': nodeIds.length,
          'edges.aggregated': 0,
          skipped: true,
          'skipped.reason': 'no-edges',
          duration_ms: Math.round(performance.now() - startMs),
        },
      };
    }
    const clusters = CommunityService.detectCommunities(nodeIds, aggregatedEdges);

    // 2. Single round-trip: match by member_set_signature + load snapshots of
    //    all existing communities in the graph (for superset matching in TS
    //    and for namer-context in the first-pass naming below).
    const { matchesByClusterIndex, existing } = await this.communityRepo.matchClusters(
      graphId,
      clusters,
    );

    // 3. Fetch member summaries for every cluster (drift hashing for matched
    //    sets, delta extraction for supersets, full tournament input for the
    //    rest). One batch over the union of cluster members.
    const allMemberIds = new Set<Uuid>();
    for (const cluster of clusters) {
      for (const id of cluster) allMemberIds.add(id);
    }
    const summaryRows =
      allMemberIds.size > 0
        ? await this.entityNodeRepo.findSummariesByIds([...allMemberIds])
        : [];
    const summaryById = new Map(summaryRows.map((r) => [r.id, r.summary]));

    // 4. Decide per-cluster route
    const routes = planRoutes(clusters, matchesByClusterIndex, existing, summaryById);

    // 5. Reduce + name in parallel (no embedding, no DB writes yet). The
    //    namer is told to avoid existing community names so the first-pass
    //    output is mostly collision-free; the resolver below cleans up the
    //    residual collisions across the freshly named set.
    const existingNames = existing.map((e) => e.name);
    const pending = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      routes.map(
        (route, idx) => () => this.executeRoute(route, idx, model, existingNames, ctx),
      ),
    );

    // 6. Stale = existing communities not consumed by any matched-set or
    //    superset route. Surviving names (from clean routes) become part of
    //    the collision avoid-set; stale ones are dropped.
    const consumedExistingIds = new Set(
      routes
        .filter(
          (r): r is Extract<ClusterRoute, { kind: 'clean' | 'incremental' }> =>
            r.kind === 'clean' || r.kind === 'incremental',
        )
        .map((r) => r.communityId),
    );
    const cleanRouteIds = new Set(
      routes.filter((r) => r.kind === 'clean').map((r) => r.communityId),
    );
    const survivingNames = existing
      .filter((e) => cleanRouteIds.has(e.id))
      .map((e) => e.name);
    const staleIds = existing
      .map((e) => e.id)
      .filter((id) => !consumedExistingIds.has(id));

    // 7. Resolve collisions across the freshly-named set + surviving names.
    const collisionsResolved = await this.resolveNameCollisions(
      pending,
      survivingNames,
      model,
      ctx,
    );

    // 8. Embed all final names in parallel.
    const toEmbed = pending.filter(
      (p): p is Extract<PendingRoute, { kind: 'full' | 'incremental' }> =>
        p.kind === 'full' || p.kind === 'incremental',
    );
    const embeddings = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      toEmbed.map((p) => () => this.embeddingService.embedText(p.name)),
    );
    const embeddingByTempId = new Map<number, number[] | null>();
    for (let i = 0; i < toEmbed.length; i++) {
      embeddingByTempId.set(toEmbed[i].tempId, embeddings[i]);
    }

    // 9. Persist: bulk insert full routes, per-row update incremental routes,
    //    delete stale.
    const freshlyCreated: Community[] = [];
    for (const p of pending) {
      if (p.kind === 'full') {
        freshlyCreated.push(
          createCommunity({
            name: NodeNameSchema.parse(p.name),
            graphId,
            summary: p.summary,
            nameEmbedding: embeddingByTempId.get(p.tempId) ?? null,
            memberIds: p.memberIds,
          }),
        );
      } else if (p.kind === 'incremental') {
        await this.communityRepo.applyIncrementalUpdate({
          id: p.communityId,
          memberIds: p.finalMemberIds,
          name: NodeNameSchema.parse(p.name),
          summary: p.summary,
          nameEmbedding: embeddingByTempId.get(p.tempId) ?? null,
        });
      }
    }
    if (freshlyCreated.length > 0) {
      await this.communityRepo.saveBulk(freshlyCreated);
    }
    if (staleIds.length > 0) {
      await this.communityRepo.deleteByIds(staleIds);
    }

    return {
      metrics: {
        ...baseMetrics,
        'nodes.count': nodeIds.length,
        'edges.aggregated': aggregatedEdges.length,
        'clusters.count': clusters.length,
        'communities.existing': existing.length,
        'communities.matched-clean': routes.filter((r) => r.kind === 'clean').length,
        'communities.matched-incremental': routes.filter((r) => r.kind === 'incremental')
          .length,
        'communities.full': routes.filter((r) => r.kind === 'full').length,
        'communities.skipped': routes.filter((r) => r.kind === 'skip').length,
        'communities.deleted': staleIds.length,
        'communities.colliders-resolved': collisionsResolved,
        duration_ms: Math.round(performance.now() - startMs),
      },
    };
  }

  /**
   * Run a single planned route through the reduce-and-name phase. Clean,
   * skip routes return noop; full/incremental routes return a pending result
   * for the post-pass collision resolver and DB writers to consume.
   */
  private async executeRoute(
    route: ClusterRoute,
    tempId: number,
    model: BaseChatModel,
    existingNames: readonly string[],
    ctx?: LlmContext,
  ): Promise<PendingRoute> {
    switch (route.kind) {
      case 'clean':
      case 'skip':
        return { kind: 'noop' };
      case 'incremental': {
        const { name, summary } = await this.buildCommunityIncremental(
          model,
          route.existingSummary,
          route.deltaSummaries,
          existingNames,
          ctx,
        );
        return {
          kind: 'incremental',
          tempId,
          communityId: route.communityId,
          name,
          summary,
          finalMemberIds: route.finalMemberIds,
        };
      }
      case 'full': {
        const { name, summary } = await this.buildCommunitySummary(
          model,
          route.memberSummaries,
          existingNames,
          ctx,
        );
        return { kind: 'full', tempId, name, summary, memberIds: route.memberIds };
      }
    }
  }

  /**
   * Incremental community update for a single entity, ported from graphiti's
   * update_community (community_operations.py:259-352) for our memberIds[]
   * storage.
   *
   * Both signature columns are refreshed by the communities trigger because
   * applyIncrementalUpdate writes member_ids in its SET list, so a touched
   * community can be matched (and skipped) by the next buildCommunities run
   * as long as no other member has drifted since.
   */
  async updateCommunityForEntity(
    userId: Uuid,
    graphId: Uuid,
    entityId: Uuid,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.updateCommunityForEntityImpl(userId, graphId, entityId, ctx);
  }

  @Span('updateCommunityForEntity', { onResult: metricsOnResult })
  private async updateCommunityForEntityImpl(
    userId: Uuid,
    graphId: Uuid,
    entityId: Uuid,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const startMs = performance.now();
    const baseMetrics: SpanMetrics = {
      'user.id': userId,
      'session.id': ctx?.sessionId,
      'graph.id': graphId,
      'entity.id': entityId,
    };

    const resolved = await this.resolveTargetCommunity(graphId, entityId);
    if ('skipMetrics' in resolved) {
      return {
        metrics: {
          ...baseMetrics,
          ...resolved.skipMetrics,
          skipped: true,
          duration_ms: Math.round(performance.now() - startMs),
        },
      };
    }
    const { target, alreadyMember, neighborCount, candidateCount } = resolved;

    const summaries = await this.entityNodeRepo.findSummariesByIds([entityId]);
    const entitySummary = summaries[0]?.summary;
    if (!entitySummary) {
      return {
        metrics: {
          ...baseMetrics,
          'community.id': target.id,
          'already.member': alreadyMember,
          'neighbors.count': neighborCount,
          'candidates.count': candidateCount,
          skipped: true,
          'skipped.reason': 'entity-deleted-or-empty',
          duration_ms: Math.round(performance.now() - startMs),
        },
      };
    }
    const model = await this.llmService.getActiveModel(userId);

    const newSummary = await this.summarizePair(
      model,
      [target.summary, entitySummary],
      ctx,
    );
    const allNames = await this.communityRepo.findNamesByGraphId(graphId);
    // Exclude the target's own name - it's about to be replaced.
    const existingNames = allNames.filter((n) => n !== target.name);
    const newName = await this.generateCommunityName(
      model,
      newSummary,
      existingNames,
      [target.summary, entitySummary],
      ctx,
    );
    const newNameEmbedding = await this.embeddingService.embedText(newName);

    const finalMemberIds = alreadyMember
      ? target.memberIds
      : [...target.memberIds, entityId];

    await this.communityRepo.applyIncrementalUpdate({
      id: target.id,
      memberIds: finalMemberIds,
      name: NodeNameSchema.parse(newName),
      summary: newSummary,
      nameEmbedding: newNameEmbedding,
    });

    return {
      metrics: {
        ...baseMetrics,
        'community.id': target.id,
        'community.memberCount': finalMemberIds.length,
        'already.member': alreadyMember,
        'neighbors.count': neighborCount,
        'candidates.count': candidateCount,
        duration_ms: Math.round(performance.now() - startMs),
      },
    };
  }

  /**
   * Mirrors graphiti's build_community (community_operations.py:174-213):
   * hierarchical pairwise reduction -> final summary -> derive name. Embedding
   * happens later, in the post-route collision-resolved pass.
   */
  @Span()
  private async buildCommunitySummary(
    model: BaseChatModel,
    entitySummaries: string[],
    existingNames: readonly string[],
    ctx?: LlmContext,
  ): Promise<{ name: string; summary: string }> {
    return this.reduceAndFinalize(model, entitySummaries, existingNames, ctx);
  }

  /**
   * Incremental seed: the existing community summary becomes the first input
   * alongside the changed-or-new members' fresh summaries. For 1 drifted
   * member this is exactly one summarize-pair + one name call; for K deltas
   * it's O(log K) rounds vs the full tournament's O(log N).
   */
  @Span()
  private async buildCommunityIncremental(
    model: BaseChatModel,
    existingSummary: string,
    deltaSummaries: string[],
    existingNames: readonly string[],
    ctx?: LlmContext,
  ): Promise<{ name: string; summary: string }> {
    return this.reduceAndFinalize(
      model,
      [existingSummary, ...deltaSummaries],
      existingNames,
      ctx,
    );
  }

  private async reduceAndFinalize(
    model: BaseChatModel,
    entitySummaries: string[],
    existingNames: readonly string[],
    ctx?: LlmContext,
  ): Promise<{ name: string; summary: string }> {
    let summaries = entitySummaries.length === 0 ? [''] : [...entitySummaries];
    let lastPairInputs: readonly [string, string] | undefined;

    while (summaries.length > 1) {
      if (summaries.length === 2) {
        lastPairInputs = [summaries[0], summaries[1]];
      }
      const odd = summaries.length % 2 === 1 ? summaries.pop()! : null;
      const half = summaries.length / 2;
      const pairs: Array<[string, string]> = [];
      for (let i = 0; i < half; i++) {
        pairs.push([summaries[i], summaries[half + i]]);
      }

      summaries = await withConcurrency(
        LLM_CONCURRENCY_LIMIT,
        pairs.map((pair) => () => this.summarizePair(model, pair, ctx)),
      );
      if (odd !== null) summaries.push(odd);
    }
    const summary = truncateAtSentence(summaries[0] ?? '', MAX_SUMMARY_CHARS);
    const name = await this.generateCommunityName(
      model,
      summary,
      existingNames,
      lastPairInputs ?? [summary, summary], // type guarantee for N<=1; Louvain drops singletons upstream
      ctx,
    );
    return { name, summary };
  }

  @Span()
  private async summarizePair(
    model: BaseChatModel,
    pair: [string, string],
    ctx?: LlmContext,
  ): Promise<string> {
    const messages = buildSummarizePairMessages({ summaries: pair });

    const { summary } = await invokeStructured(model, SummarySchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'community.summarize-pair',
      tags: ['knowledge-graph', 'community', 'summarize-pair'],
    });
    return truncateAtSentence(summary, MAX_SUMMARY_CHARS);
  }

  @Span()
  private async generateCommunityName(
    model: BaseChatModel,
    summary: string,
    existingNames: readonly string[],
    sections: readonly [string, string],
    ctx?: LlmContext,
  ): Promise<string> {
    const messages = buildCommunityNameMessages({ summary, existingNames, sections });
    const { name } = await invokeStructured(model, CommunityNameSchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'community.community-name',
      tags: ['knowledge-graph', 'community', 'community-name'],
    });
    return name;
  }

  /**
   * Detect collisions across freshly-named routes + surviving (clean-route)
   * names. Groups with size > 1 (or any fresh name matching a surviving one)
   * are sent to a single batched LLM call that renames just the colliders.
   * Returns the number of colliders that were renamed.
   */
  @Span()
  private async resolveNameCollisions(
    pending: PendingRoute[],
    survivingNames: readonly string[],
    model: BaseChatModel,
    ctx?: LlmContext,
  ): Promise<number> {
    const fresh = pending.filter(
      (p): p is Extract<PendingRoute, { kind: 'full' | 'incremental' }> =>
        p.kind === 'full' || p.kind === 'incremental',
    );
    if (fresh.length === 0) return 0;

    const survivingNorm = new Set(survivingNames.map(normalizeName));
    const freshByNorm = new Map<string, typeof fresh>();
    for (const p of fresh) {
      const k = normalizeName(p.name);
      const bucket = freshByNorm.get(k) ?? [];
      bucket.push(p);
      freshByNorm.set(k, bucket);
    }

    const colliderTempIds = new Set<number>();
    for (const [norm, bucket] of freshByNorm) {
      // Any fresh group with >1 members is a collision; even a single fresh
      // entry colliding with a surviving name needs renaming.
      if (bucket.length > 1 || survivingNorm.has(norm)) {
        for (const p of bucket) colliderTempIds.add(p.tempId);
      }
    }
    if (colliderTempIds.size === 0) return 0;

    const colliders: Collider[] = fresh
      .filter((p) => colliderTempIds.has(p.tempId))
      .map((p) => ({ tempId: p.tempId, summary: p.summary }));

    const survivingPlusNonCollider = [
      ...survivingNames,
      ...fresh.filter((p) => !colliderTempIds.has(p.tempId)).map((p) => p.name),
    ];

    const validatorCtx = { colliders, namesInUse: survivingPlusNonCollider };
    const messages = buildResolveNameCollisionsMessages(validatorCtx);
    const { resolutions } = await invokeStructured(
      model,
      ResolveNameCollisionsSchema,
      messages,
      {
        callbacks: this.llmTracer.getCallbacks(ctx),
        runName: 'community.resolve-name-collisions',
        tags: ['knowledge-graph', 'community', 'resolve-name-collisions'],
        validate: buildResolveNameCollisionsValidator(validatorCtx),
      },
    );

    const freshByTempId = new Map(fresh.map((p) => [p.tempId, p]));
    for (const r of resolutions) {
      freshByTempId.get(r.tempId)!.name = r.name;
    }
    return colliderTempIds.size;
  }

  /**
   * Build the weighted undirected graph from node ids + Postgres-aggregated
   * edges and partition it into communities via Louvain modularity
   * optimization. Parallel facts between a pair raise their edge weight, so
   * densely-linked entities cluster together. Singleton clusters (isolated
   * nodes or one-member partitions) are dropped - a one-entity community wastes
   * an LLM summarize/embed and conveys nothing. Returned clusters have sorted
   * members for stable signatures.
   */
  private static detectCommunities(
    nodeIds: Uuid[],
    aggregatedEdges: Array<{ a: Uuid; b: Uuid; edgeCount: number }>,
  ): Uuid[][] {
    const graph = new UndirectedGraph();

    for (const id of nodeIds) graph.addNode(id);

    for (const { a, b, edgeCount } of aggregatedEdges) {
      if (!graph.hasNode(a) || !graph.hasNode(b)) continue;
      graph.addEdge(a, b, { weight: edgeCount });
    }

    const communityByNode = louvain(graph, {
      getEdgeWeight: 'weight',
      rng: mulberry32(LOUVAIN_SEED),
    });
    const byCommunity = new Map<number, Uuid[]>();

    for (const [nodeId, community] of Object.entries(communityByNode)) {
      const members = byCommunity.get(community) ?? [];
      members.push(nodeId as Uuid);
      byCommunity.set(community, members);
    }
    return [...byCommunity.values()]
      .filter((members) => members.length > 1)
      .map((members) => members.sort());
  }

  /**
   * Resolve the community an entity should belong to: its existing membership
   * if any, otherwise the mode of its neighbors' communities. Returns
   * `skipMetrics` (counts + reason) when no target can be determined.
   */
  private async resolveTargetCommunity(
    graphId: Uuid,
    entityId: Uuid,
  ): Promise<
    | {
        target: Community;
        alreadyMember: boolean;
        neighborCount: number;
        candidateCount: number;
      }
    | { skipMetrics: SpanMetrics }
  > {
    const existing = await this.communityRepo.findByMemberId(graphId, entityId);
    if (existing) {
      return {
        target: existing,
        alreadyMember: true,
        neighborCount: 0,
        candidateCount: 0,
      };
    }

    const neighborIds = await this.entityEdgeRepo.findNeighborIds(entityId);
    const neighborCount = neighborIds.length;
    if (neighborCount === 0) {
      return {
        skipMetrics: { 'neighbors.count': 0, 'skipped.reason': 'no-neighbors' },
      };
    }

    const candidates = await this.communityRepo.findByAnyMember(graphId, neighborIds);
    const candidateCount = candidates.length;
    if (candidateCount === 0) {
      return {
        skipMetrics: {
          'neighbors.count': neighborCount,
          'candidates.count': 0,
          'skipped.reason': 'no-candidate-communities',
        },
      };
    }

    const mode = CommunityService.pickModeCommunity(neighborIds, candidates);
    if (!mode) {
      return {
        skipMetrics: {
          'neighbors.count': neighborCount,
          'candidates.count': candidateCount,
          'skipped.reason': 'no-mode-community',
        },
      };
    }
    return { target: mode, alreadyMember: false, neighborCount, candidateCount };
  }

  /**
   * Pick the community with the most neighbor members. Tiebreak by largest
   * community id for a deterministic tiebreak.
   */
  private static pickModeCommunity(
    neighborIds: Uuid[],
    candidates: Community[],
  ): Community | null {
    if (candidates.length === 0) return null;
    const neighborSet = new Set(neighborIds);

    const scored = candidates.map((c) => ({
      community: c,
      count: c.memberIds.filter((id) => neighborSet.has(id)).length,
    }));
    scored.sort((x, y) => {
      if (x.count !== y.count) return y.count - x.count;
      return x.community.id < y.community.id ? 1 : -1;
    });
    return scored[0].count > 0 ? scored[0].community : null;
  }
}
