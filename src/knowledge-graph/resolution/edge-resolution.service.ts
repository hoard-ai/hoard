import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

import type { Uuid } from '@/common/schemas';
import { invokeStructured } from '@/llm';
import {
  LLM_TRACER,
  type LlmContext,
  type LlmTracer,
  metricsOnResult,
  Span,
  type SpanMetrics,
} from '@/observability';

import { compressIdMap, LLM_CONCURRENCY_LIMIT, withConcurrency } from '../batch-utils';
import { EntityEdge, EpisodicNode } from '../models';
import {
  buildDedupeEdgesMessages,
  buildDedupeEdgesValidator,
  EdgeDedupeSchema,
} from '../prompts';
import { selectChunkText } from '../prompts/text-utils';
import { EntityEdgeRepository } from '../repository/repositories';
import { SearchBySimilarityParamsSchema, SearchByTextParamsSchema } from '../types';
import {
  CANDIDATE_LIMIT,
  cosineSimilarity,
  FACT_SIMILARITY_THRESHOLD,
  MAX_CANDIDATES,
  MAX_KEYWORD_CANDIDATES,
  normalizeString,
} from './resolution-utils';
import { type DedupeEdgesResult, type EdgeChunkSources } from './types';

@Injectable()
export class EdgeResolutionService {
  constructor(
    private readonly edgeRepo: EntityEdgeRepository,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  /* Builds a synthetic episode whose content is scoped to the chunks the edge
   * came from - resolved against its ORIGIN episode (chunkSources is qualified by
   * episodeIndex), so a cross-episode-merged edge uses the right chunk array. */
  private episodeForChunks(
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    edgeId: Uuid,
  ): EpisodicNode {
    const source = chunkSources.get(edgeId);
    if (!source) {
      throw new Error(
        `episodeForChunks: edge ${edgeId} has no originating chunk indices`,
      );
    }
    const { episodeIndex, indices } = source;
    return {
      ...episodes[episodeIndex],
      content: selectChunkText(indices, chunksPerEpisode[episodeIndex]),
    };
  }

  async collectCandidates(edges: EntityEdge[], graphId: Uuid): Promise<EntityEdge[]> {
    const { candidates } = await this.collectCandidatesImpl(edges, graphId);
    return candidates;
  }

  @Span('collectEdgeCandidates', {
    observationKind: 'retriever',
    onResult: metricsOnResult,
  })
  private async collectCandidatesImpl(
    edges: EntityEdge[],
    graphId: Uuid,
  ): Promise<{ candidates: EntityEdge[]; metrics: SpanMetrics }> {
    // Same-endpoint edges (`getBetweenNodes`) are fetched explicitly per edge:
    // text + similarity searches may not surface an existing edge whose fact
    // differs textually from the new one, but a duplicate or contradiction
    // between the same two nodes still needs to be considered during dedup.
    // Mirrors upstream `EntityEdge.get_between_nodes` in edge_operations.py.
    const results = await Promise.all(
      edges.flatMap((e) => [
        this.edgeRepo.searchByFact(
          SearchByTextParamsSchema.parse({
            query: e.fact,
            graphIds: [graphId],
            limit: CANDIDATE_LIMIT,
          }),
        ),
        e.factEmbedding !== null
          ? this.edgeRepo.searchBySimilarity(
              SearchBySimilarityParamsSchema.parse({
                embedding: e.factEmbedding,
                graphIds: [graphId],
                limit: CANDIDATE_LIMIT,
              }),
            )
          : Promise.resolve([] as EntityEdge[]),
        this.edgeRepo.getBetweenNodes(e.sourceNodeId, e.targetNodeId),
      ]),
    );
    const seen = new Set<Uuid>();
    const candidates = results.flat().filter((e) => {
      if (seen.has(e.id)) return false;
      seen.add(e.id);
      return true;
    });
    return {
      candidates,
      metrics: {
        'input.count': edges.length,
        'graph.id': graphId,
        'candidates.count': candidates.length,
      },
    };
  }

  /**
   * Dedup pass (vs the live graph). Runs the `dedupe-edges` LLM comparison for
   * each extracted edge and partitions the results:
   * - `matchedExistingEdges`: existing graph edges an extracted edge duplicated,
   *   with the new episode(s) appended (re-saved as-is, never enriched).
   * - `survivors`: freshly extracted edges with no duplicate (= newEdges).
   * - `contradictionsBySurvivorId`: per survivor, the existing edges it
   *   contradicts
   */
  async dedupeEdges(
    model: BaseChatModel,
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    extractedEdges: EntityEdge[],
    idMap: Map<Uuid, Uuid>,
    referenceTime: Date,
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<DedupeEdgesResult> {
    const { metrics: _m, ...rest } = await this.dedupeEdgesImpl(
      model,
      episodes,
      chunksPerEpisode,
      chunkSources,
      extractedEdges,
      idMap,
      referenceTime,
      previousEpisodes,
      customInstructions,
      ctx,
    );
    return rest;
  }

  @Span('dedupeEdges', { onResult: metricsOnResult })
  private async dedupeEdgesImpl(
    model: BaseChatModel,
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    extractedEdges: EntityEdge[],
    idMap: Map<Uuid, Uuid>,
    referenceTime: Date,
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<DedupeEdgesResult & { metrics: SpanMetrics }> {
    // Step 1: Remap source/target ids via idMap
    const remapped = extractedEdges.map((e) => ({
      ...e,
      sourceNodeId: idMap.get(e.sourceNodeId) ?? e.sourceNodeId,
      targetNodeId: idMap.get(e.targetNodeId) ?? e.targetNodeId,
    }));
    // dedupeEdges is invoked per origin episode, so every edge here shares one
    // graphId. collectCandidates and the per-edge searches below rely on this
    if (new Set(remapped.map((e) => e.graphId)).size > 1) {
      throw new Error('dedupeEdges: edges span multiple graphIds');
    }

    // Candidates use the remapped (canonical) endpoints for getBetweenNodes.
    const existingEdges = remapped.length
      ? await this.collectCandidates(remapped, remapped[0].graphId)
      : [];

    // Step 2: Intra-batch dedup - same endpoints + same normalized fact → keep first, merge episodes
    const deduped: EntityEdge[] = [];
    for (const edge of remapped) {
      const normalizedFact = normalizeString(edge.fact);
      const existing = deduped.find(
        (d) =>
          d.sourceNodeId === edge.sourceNodeId &&
          d.targetNodeId === edge.targetNodeId &&
          normalizeString(d.fact) === normalizedFact,
      );
      if (existing) {
        // Merge episodes into the first occurrence. Chunk sources stay keyed by
        // their own edge id (qualified by origin episode), so the kept edge
        // selects its own episode text - no union needed.
        for (const ep of edge.episodes) {
          if (!existing.episodes.includes(ep)) {
            existing.episodes.push(ep);
          }
        }
      } else {
        deduped.push(edge);
      }
    }

    const matchedExistingEdges: EntityEdge[] = [];
    const survivors: EntityEdge[] = [];
    const resolvedExistingIds = new Set<Uuid>();
    const contradictionsBySurvivorId = new Map<Uuid, EntityEdge[]>();

    for (const edge of deduped) {
      // Find same-endpoint existing edges (same direction only). Reversed-direction
      // duplicates are left to cosine/keyword retrieval to surface as similar-topic
      // candidates - the prompt does not reason about endpoint direction.
      const endpointEdges = existingEdges.filter(
        (e) =>
          e.sourceNodeId === edge.sourceNodeId && e.targetNodeId === edge.targetNodeId,
      );

      // Find similar-fact edges (cosine) excluding same-endpoint already found
      const endpointIds = new Set(endpointEdges.map((e) => e.id));

      // Cosine candidates (in-memory)
      const cosineEdges: EntityEdge[] =
        edge.factEmbedding !== null
          ? existingEdges
              .filter((e) => !endpointIds.has(e.id) && e.factEmbedding !== null)
              .map((e) => ({
                edge: e,
                score: cosineSimilarity(edge.factEmbedding!, e.factEmbedding!),
              }))
              .filter((s) => s.score >= FACT_SIMILARITY_THRESHOLD)
              .sort((a, b) => b.score - a.score)
              .slice(0, MAX_CANDIDATES)
              .map((s) => s.edge)
          : [];

      // Keyword candidates (BM25 fulltext)
      const keywordEdges = await this.edgeRepo.searchByFact(
        SearchByTextParamsSchema.parse({
          query: edge.fact,
          graphIds: [edge.graphId],
          limit: MAX_KEYWORD_CANDIDATES,
        }),
      );

      // Merge: cosine-first, then keyword-only additions (deduped, endpoint-excluded)
      const cosineIds = new Set(cosineEdges.map((e) => e.id));
      const keywordOnly = keywordEdges.filter(
        (e) => !endpointIds.has(e.id) && !cosineIds.has(e.id),
      );
      const similarEdges: EntityEdge[] = [...cosineEdges, ...keywordOnly];

      if (endpointEdges.length === 0 && similarEdges.length === 0) {
        survivors.push(edge);
        continue;
      }

      const { dedupe, idxToEdge } = await this.dedupeEdgeViaLlm(
        model,
        edge,
        endpointEdges,
        similarEdges,
        this.episodeForChunks(episodes, chunksPerEpisode, chunkSources, edge.id),
        previousEpisodes,
        referenceTime,
        customInstructions,
        ctx,
      );
      const isDuplicate = dedupe.duplicateFacts.length > 0;

      if (isDuplicate) {
        // Append the resolved edge's originating episode(s) to the matching
        // existing endpoint edge(s) and include them so they are re-saved with
        // updated episodes. Using edge.episodes (not a single batch episode)
        // stays correct when the list holds a cross-episode-merged edge.
        // Mirrors Python edge_operations.py:523-524 and 581-582.
        //
        // NOTE (accepted behavior): duplicates are NOT adjudicated for
        // contradictions. A new edge that both duplicates an existing edge AND
        // contradicts a third edge does not re-invalidate the third edge here
        for (const idx of dedupe.duplicateFacts) {
          const existingEdge = idxToEdge.get(idx)!;

          if (resolvedExistingIds.has(existingEdge.id)) continue;
          for (const ep of edge.episodes) {
            if (!existingEdge.episodes.includes(ep)) existingEdge.episodes.push(ep);
          }
          matchedExistingEdges.push(existingEdge);
          resolvedExistingIds.add(existingEdge.id);
        }
      } else {
        survivors.push(edge);
        contradictionsBySurvivorId.set(
          edge.id,
          dedupe.contradictedFacts.map((idx) => idxToEdge.get(idx)!),
        );
      }
    }

    return {
      matchedExistingEdges,
      survivors,
      contradictionsBySurvivorId,
      metrics: {
        'episodes.count': episodes.length,
        'extracted.count': extractedEdges.length,
        'existing.count': existingEdges.length,
        'matched.count': matchedExistingEdges.length,
        'survivors.count': survivors.length,
      },
    };
  }

  /**
   * Temporal invalidation over the enriched survivors. Pure arithmetic over the
   * now-filled validAt/invalidAt (no model call). Consumes the contradictions
   * recorded by `dedupeEdges`. Runs GLOBALLY over the whole batch's survivors so
   * an existing edge contradicted by survivors in two episodes is invalidated
   * once (first-survivor-wins); `invalidatedBySurvivorId` lets the orchestrator
   * attribute each invalidation back to its origin episode.
   */
  invalidateEdges(
    survivors: EntityEdge[],
    contradictionsBySurvivorId: Map<Uuid, EntityEdge[]>,
  ): {
    invalidatedEdges: EntityEdge[];
    invalidatedBySurvivorId: Map<Uuid, EntityEdge[]>;
  } {
    const { metrics: _m, ...rest } = this.invalidateEdgesImpl(
      survivors,
      contradictionsBySurvivorId,
    );
    return rest;
  }

  @Span('invalidateEdges', { onResult: metricsOnResult })
  private invalidateEdgesImpl(
    survivors: EntityEdge[],
    contradictionsBySurvivorId: Map<Uuid, EntityEdge[]>,
  ): {
    invalidatedEdges: EntityEdge[];
    invalidatedBySurvivorId: Map<Uuid, EntityEdge[]>;
    metrics: SpanMetrics;
  } {
    const invalidatedEdgesMap = new Map<Uuid, EntityEdge>();
    const invalidatedBySurvivorId = new Map<Uuid, EntityEdge[]>();

    for (const survivor of survivors) {
      const contradictions = contradictionsBySurvivorId.get(survivor.id) ?? [];

      // Guard (a): the new edge already carries an end - mark it expired now.
      if (survivor.invalidAt && !survivor.expiredAt) {
        survivor.expiredAt = new Date();
      }

      // Guard (b): self-expiration - if any contradiction candidate postdates
      // this edge, the edge is superseded by information already in the graph.
      if (!survivor.expiredAt && survivor.validAt !== null) {
        const contradictionCandidates = contradictions
          .filter((c) => c.validAt !== null)
          .sort((a, b) => a.validAt!.getTime() - b.validAt!.getTime());
        for (const candidate of contradictionCandidates) {
          if (candidate.validAt! > survivor.validAt) {
            survivor.invalidAt = candidate.validAt;
            survivor.expiredAt = new Date();
            break;
          }
        }
      }

      // Guard (c): invalidate existing edges that genuinely overlap with the new
      // edge's validity window and predate it. Mirrors Python
      // resolve_edge_contradictions (edge_operations.py:425-460).
      for (const existing of contradictions) {
        if (invalidatedEdgesMap.has(existing.id)) continue;

        const edgeInvalidAt = existing.invalidAt;
        const resolvedValidAt = survivor.validAt;
        const edgeValidAt = existing.validAt;
        const resolvedInvalidAt = survivor.invalidAt;

        // Skip if there is no temporal overlap between the two edges.
        if (
          (edgeInvalidAt !== null &&
            resolvedValidAt !== null &&
            edgeInvalidAt <= resolvedValidAt) ||
          (edgeValidAt !== null &&
            resolvedInvalidAt !== null &&
            resolvedInvalidAt <= edgeValidAt)
        )
          continue;

        // Only invalidate if the existing edge predates the new edge.
        if (
          edgeValidAt !== null &&
          resolvedValidAt !== null &&
          edgeValidAt < resolvedValidAt
        ) {
          const invalidated: EntityEdge = {
            ...existing,
            invalidAt: survivor.validAt,
            expiredAt: existing.expiredAt ?? new Date(),
          };
          invalidatedEdgesMap.set(existing.id, invalidated);
          const list = invalidatedBySurvivorId.get(survivor.id) ?? [];
          list.push(invalidated);
          invalidatedBySurvivorId.set(survivor.id, list);
        }
      }
    }
    const invalidatedEdges = Array.from(invalidatedEdgesMap.values());

    return {
      invalidatedEdges,
      invalidatedBySurvivorId,
      metrics: {
        'survivors.count': survivors.length,
        'invalidated.count': invalidatedEdges.length,
      },
    };
  }

  // Cross-batch edge dedup. Mirrors upstream `dedupe_edges_bulk`
  // (bulk_utils.py:489): for each batch edge, surface peer edges from other
  // episodes in the same batch as candidates and let the LLM identify
  // duplicates. Without this, two episodes mentioning the same fact would each
  // persist a separate row because per-episode resolution only consults the
  // live graph. Returns the flat set of distinct canonical edges (collapsed
  // duplicates dropped) whose `episodes` field is the union of the originating
  // episode IDs. The caller re-partitions by origin episode for resolution.
  async dedupeAcrossBatch(
    model: BaseChatModel,
    edgesPerEpisode: EntityEdge[][],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    previousEpisodesPerEpisode: EpisodicNode[][],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<EntityEdge[]> {
    return this.dedupeAcrossBatchImpl(
      model,
      edgesPerEpisode,
      episodes,
      chunksPerEpisode,
      chunkSources,
      previousEpisodesPerEpisode,
      customInstructions,
      ctx,
    ).then((r) => r.canonicalEdges);
  }

  @Span('dedupeAcrossBatch', { onResult: metricsOnResult })
  private async dedupeAcrossBatchImpl(
    model: BaseChatModel,
    edgesPerEpisode: EntityEdge[][],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    previousEpisodesPerEpisode: EpisodicNode[][],
    customInstructions: string | undefined,
    ctx: LlmContext | undefined,
  ): Promise<{ canonicalEdges: EntityEdge[]; metrics: SpanMetrics }> {
    const allEdges = edgesPerEpisode.flat();
    const baseMetrics: SpanMetrics = {
      'episodes.count': episodes.length,
      'edges.in': allEdges.length,
    };

    if (allEdges.length < 2) {
      return {
        canonicalEdges: allEdges,
        metrics: { ...baseMetrics, 'pairs.found': 0 },
      };
    }

    // Owner index: which episode each edge came from. Edge IDs are unique
    // (factory generates randomUUID per extraction), so a Map keyed by id
    // is unambiguous.
    const edgeOwner = new Map<Uuid, number>();
    edgesPerEpisode.forEach((edges, i) => {
      for (const e of edges) edgeOwner.set(e.id, i);
    });

    type Task = {
      edge: EntityEdge;
      endpointEdges: EntityEdge[];
      similarEdges: EntityEdge[];
    };
    const tasks: Task[] = [];
    for (const edge of allEdges) {
      const endpointEdges: EntityEdge[] = [];
      const similarEdges: EntityEdge[] = [];
      for (const peer of allEdges) {
        if (peer.id === edge.id) continue;
        const sameEndpoints =
          peer.sourceNodeId === edge.sourceNodeId &&
          peer.targetNodeId === edge.targetNodeId;
        if (sameEndpoints) {
          endpointEdges.push(peer);
          continue;
        }
        if (
          edge.factEmbedding !== null &&
          peer.factEmbedding !== null &&
          cosineSimilarity(edge.factEmbedding, peer.factEmbedding) >=
            FACT_SIMILARITY_THRESHOLD
        ) {
          similarEdges.push(peer);
        }
      }
      if (endpointEdges.length === 0 && similarEdges.length === 0) continue;
      tasks.push({
        edge,
        endpointEdges,
        similarEdges: similarEdges.slice(0, MAX_CANDIDATES),
      });
    }

    const pairResults = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      tasks.map((t) => async (): Promise<[Uuid, Uuid][]> => {
        const ownerIdx = edgeOwner.get(t.edge.id)!;
        const { dedupe, idxToEdge } = await this.dedupeEdgeViaLlm(
          model,
          t.edge,
          t.endpointEdges,
          t.similarEdges,
          this.episodeForChunks(episodes, chunksPerEpisode, chunkSources, t.edge.id),
          previousEpisodesPerEpisode[ownerIdx],
          episodes[ownerIdx].validAt,
          customInstructions,
          ctx,
        );
        // Only endpoint-range indices (same + reversed) count as duplicates.
        // The similar-topic section is for contradictions; accepting duplicates
        // from it would collapse edges with different endpoints. Matches the
        // guard in `dedupeEdges` and upstream `dedupe_edges_bulk` semantics
        // (bulk_utils.py:521-524) which never surfaces non-endpoint duplicates.
        const endpointCount = t.endpointEdges.length;
        const localPairs: [Uuid, Uuid][] = [];
        for (const idx of dedupe.duplicateFacts) {
          if (idx >= endpointCount) continue;
          const peer = idxToEdge.get(idx);
          if (peer) localPairs.push([t.edge.id, peer.id]);
        }
        return localPairs;
      }),
    );

    const duplicatePairs = pairResults.flat();
    if (duplicatePairs.length === 0) {
      return {
        canonicalEdges: allEdges,
        metrics: { ...baseMetrics, 'pairs.found': 0, 'edges.out': allEdges.length },
      };
    }

    // Union-find collapses transitive duplicates and picks lex-smallest ID as
    // canonical. canonicalById holds every edge that is its own canonical -
    // merge winners and untouched singletons alike (an edge absent from idMap
    // maps to itself); the merged-away losers are simply excluded.
    const idMap = compressIdMap<Uuid>(duplicatePairs);
    const canonicalById = new Map<Uuid, EntityEdge>();

    for (const edge of allEdges) {
      const canonicalId = idMap.get(edge.id) ?? edge.id;
      if (canonicalId === edge.id) {
        canonicalById.set(canonicalId, edge);
      }
    }
    for (const edge of allEdges) {
      const canonicalId = idMap.get(edge.id) ?? edge.id;
      if (canonicalId === edge.id) continue;
      const canonical = canonicalById.get(canonicalId);
      if (!canonical) continue;
      for (const ep of edge.episodes) {
        if (!canonical.episodes.includes(ep)) canonical.episodes.push(ep);
      }
      // No chunk-source union: each edge id keeps its own origin-episode-qualified
      // entry, and the canonical edge selects its own episode text downstream.
    }

    const canonicalEdges = [...canonicalById.values()];

    return {
      canonicalEdges,
      metrics: {
        ...baseMetrics,
        'pairs.found': duplicatePairs.length,
        'edges.out': canonicalEdges.length,
      },
    };
  }

  // Shared LLM-driven dedup call used by both per-episode `dedupeEdges` and
  // batch-wide `dedupeAcrossBatch`. Builds the integer-indexed candidate list,
  // invokes the structured-output prompt, and returns the raw decisions plus
  // the idx → edge map so callers can act on duplicateFacts / contradictedFacts.
  private async dedupeEdgeViaLlm(
    model: BaseChatModel,
    edge: EntityEdge,
    endpointEdges: EntityEdge[],
    similarEdges: EntityEdge[],
    episode: EpisodicNode,
    previousEpisodes: EpisodicNode[],
    referenceTime: Date,
    customInstructions: string | undefined,
    ctx: LlmContext | undefined,
  ): Promise<{
    dedupe: { duplicateFacts: number[]; contradictedFacts: number[] };
    idxToEdge: Map<number, EntityEdge>;
  }> {
    // TODO: reversed-direction duplicates can slip through. Endpoint matching
    // is same-direction only (matches Graphiti), so a fact like "Acme employs
    // Alice" won't collide with an existing "Alice works at Acme" via the
    // endpoint bucket. It only surfaces if cosine/keyword retrieval lifts it
    // into similarEdges - and even then the duplicate guard ignores matches
    // outside the endpoint range, so the LLM can only flag it as a
    // contradiction (or miss it entirely). Revisit once we have an eval set.

    // Continuous indices: endpoint → similar. The duplicate guard in callers
    // relies on idx < endpointEdges.length, so order matters.
    const endpointWithIdx = endpointEdges.map((e, i) => ({ idx: i, edge: e }));
    const similarOffset = endpointEdges.length;
    const similarWithIdx = similarEdges.map((e, i) => ({
      idx: similarOffset + i,
      edge: e,
    }));

    const idxToEdge = new Map<number, EntityEdge>();
    for (const { idx, edge: e } of endpointWithIdx) idxToEdge.set(idx, e);
    for (const { idx, edge: e } of similarWithIdx) idxToEdge.set(idx, e);

    const messages = buildDedupeEdgesMessages({
      episode,
      previousEpisodes,
      newEdge: { name: edge.name, fact: edge.fact },
      endpointEdges: endpointWithIdx.map(({ idx, edge: e }) => ({
        idx,
        name: e.name,
        fact: e.fact,
      })),
      similarEdges: similarWithIdx.map(({ idx, edge: e }) => ({
        idx,
        name: e.name,
        fact: e.fact,
      })),
      referenceTime,
      customInstructions,
    });

    const dedupe = await invokeStructured(model, EdgeDedupeSchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'resolve-edges',
      tags: ['knowledge-graph', 'resolution.edge'],
      validate: buildDedupeEdgesValidator({
        endpointEdges: endpointWithIdx.map(({ idx, edge: e }) => ({
          idx,
          name: e.name,
          fact: e.fact,
        })),
        similarEdges: similarWithIdx.map(({ idx, edge: e }) => ({
          idx,
          name: e.name,
          fact: e.fact,
        })),
      }),
    });

    return { dedupe, idxToEdge };
  }
}
