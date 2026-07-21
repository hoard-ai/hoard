import { setMaxListeners } from 'events';

import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

import type { Uuid } from '@/common/schemas';
import { KnowledgeGraphConfigService } from '@/config/knowledge-graph';
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
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import {
  buildDirectedIdMap,
  CountingSemaphore,
  reassembleByOffsets,
  remapEdgeEndpointsToCanonical,
  withConcurrency,
} from '../batch-utils';
import { CommunityMaintenanceService } from '../community';
import { EmbeddingService } from '../embedding';
import { EdgeExtractionService, NodeExtractionService } from '../extraction';
import {
  createEpisodicEdge,
  createEpisodicNode,
  createSaga,
  EntityEdge,
  EntityNode,
  EpisodicNode,
} from '../models';
import { buildSummarizeSagasMessages, SagaSummarySchema } from '../prompts';
import {
  EntityEdgeRepository,
  EntityNodeRepository,
  EpisodicEdgeRepository,
  EpisodicNodeRepository,
  SagaRepository,
} from '../repository';
import { EdgeResolutionService, NodeResolutionService } from '../resolution';
import {
  EpisodeType,
  NodeNameSchema,
  type RetrieveEpisodesParamsInput,
  RetrieveEpisodesParamsSchema,
} from '../types';
import { prepareChunks } from './content-chunking';
import {
  buildEdgeCorefCandidates,
  buildNodeContext,
  getEffectiveTypeMappings,
  recomputeCanonicalNodesByCanonicalIdMap,
  recomputeCorefByCanonicalIdMap,
} from './episode-utils';
import {
  AddEpisodeResult,
  AddJsonEpisodesOptionsInput,
  AddJsonEpisodesOptionsSchema,
  AddMessageEpisodesOptionsInput,
  AddMessageEpisodesOptionsSchema,
  AddTextEpisodesOptionsInput,
  AddTextEpisodesOptionsSchema,
  type BatchState,
  EpisodeWorkItem,
  type NormalizedAddEpisodeOptions,
  PERSIST_TRANSACTION_MAX_WAIT_MS,
  PERSIST_TRANSACTION_TIMEOUT_MS,
  type PipelineConfig,
  PREVIOUS_EPISODES_WINDOW,
} from './types';

@Injectable()
export class EpisodeService {
  constructor(
    private readonly llmService: LlmService,
    private readonly communityMaintenance: CommunityMaintenanceService,
    private readonly embeddingService: EmbeddingService,
    private readonly nodeExtractionService: NodeExtractionService,
    private readonly edgeExtractionService: EdgeExtractionService,
    private readonly nodeResolutionService: NodeResolutionService,
    private readonly edgeResolutionService: EdgeResolutionService,
    private readonly entityNodeRepository: EntityNodeRepository,
    private readonly entityEdgeRepository: EntityEdgeRepository,
    private readonly episodicNodeRepository: EpisodicNodeRepository,
    private readonly episodicEdgeRepository: EpisodicEdgeRepository,
    private readonly sagaRepository: SagaRepository,
    private readonly prisma: PrismaService,
    private readonly kgConfig: KnowledgeGraphConfigService,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  private static makeCtx(parsed: NormalizedAddEpisodeOptions): LlmContext {
    const uniqueGraphIds = [...new Set(parsed.episodes.map((e) => e.graphId))];
    return {
      userId: parsed.userId,
      tags: [
        'knowledge-graph',
        'ingestion',
        ...uniqueGraphIds.map((id) => `graph:${id}`),
      ],
      metadata: {
        episodeCount: String(parsed.episodes.length),
      },
    };
  }

  @Span('getEpisodes')
  async getEpisodes(options: RetrieveEpisodesParamsInput): Promise<EpisodicNode[]> {
    const params = RetrieveEpisodesParamsSchema.parse(options);
    return this.episodicNodeRepository.retrieveEpisodes(params);
  }

  /**
   * TODO: Deletion is currently best-effort and leaves downstream graph state
   * inconsistent. Non-originating episodes can still mutate surviving edges
   * (invalidAt/expiredAt stamps, episodes[] arrays, node attributes); none of
   * that is unwound here. The right design is dependency-aware reconsolidation
   * on retrieval over an append-only graph, but the trade-offs only become
   * legible against a real graph with real query patterns - revisit once we
   * have one. Design notes: PLAN.md.
   */
  async deleteEpisode(userId: Uuid, id: Uuid): Promise<void> {
    await this.deleteEpisodeImpl(userId, id);
  }

  @Span('deleteEpisode', { onResult: metricsOnResult })
  private async deleteEpisodeImpl(
    userId: Uuid,
    id: Uuid,
  ): Promise<{ metrics: SpanMetrics }> {
    const episode = await this.episodicNodeRepository.getById(id);
    if (!episode) {
      return { metrics: { 'episode.id': id, skipped: true } };
    }

    // Load entity nodes mentioned by this episode
    const mentionedNodeIds = await this.episodicNodeRepository.getMentionedEntityIds(id);

    // Delete entity nodes that are only mentioned by this episode
    await Promise.all(
      mentionedNodeIds.map((nodeId) =>
        this.entityNodeRepository.deleteIfSoleMentioned(nodeId),
      ),
    );

    // Load and delete entity edges first created by this episode
    const edgeIds = await this.entityEdgeRepository.getIdsForEpisodeDeletion(id);
    if (edgeIds.length > 0) {
      await this.entityEdgeRepository.deleteByIds(edgeIds);
    }

    // Delete MENTIONS edges for this episode
    await this.episodicEdgeRepository.deleteBySourceId(id);

    // Delete episode node
    await this.episodicNodeRepository.delete(id);

    // Connectivity changed around the mentioned entities; since-deleted members skip gracefully.
    if (mentionedNodeIds.length > 0) {
      await this.communityMaintenance.scheduleMaintenance(
        userId,
        episode.graphId,
        mentionedNodeIds,
      );
    }

    return {
      metrics: {
        'episode.id': id,
        'nodes.mentioned': mentionedNodeIds.length,
        'edges.deleted': edgeIds.length,
      },
    };
  }

  /**
   * TODO: For very large batches a bulk variant would be preferred over
   * sequential per-episode deletion. (graph consistency problem though)
   */
  async deleteEpisodesById(userId: Uuid, ids: Uuid[]): Promise<void> {
    await this.deleteEpisodesByIdImpl(userId, ids);
  }

  @Span('deleteEpisodesById', { onResult: metricsOnResult })
  private async deleteEpisodesByIdImpl(
    userId: Uuid,
    ids: Uuid[],
  ): Promise<{ metrics: SpanMetrics }> {
    await Promise.all(ids.map((id) => this.deleteEpisode(userId, id)));
    return { metrics: { 'episodes.count': ids.length } };
  }

  async summarizeSaga(options: {
    userId: Uuid;
    sagaId: Uuid;
    graphId: Uuid;
  }): Promise<string> {
    const { summary } = await this.summarizeSagaImpl(options);
    return summary;
  }

  @Span('summarizeSaga', { onResult: metricsOnResult })
  private async summarizeSagaImpl(options: {
    userId: Uuid;
    sagaId: Uuid;
    graphId: Uuid;
  }): Promise<{ summary: string; metrics: SpanMetrics }> {
    const { userId, sagaId, graphId } = options;
    const ctx: LlmContext = {
      userId,
      tags: ['knowledge-graph', 'saga'],
      metadata: { sagaId, graphId },
    };

    const baseMetrics: SpanMetrics = {
      'user.id': ctx.userId,
      'session.id': ctx.sessionId,
      'saga.id': sagaId,
      'graph.id': graphId,
    };

    const model = await this.llmService.getActiveModel(userId);

    const saga = await this.sagaRepository.getById(sagaId);
    if (!saga) {
      throw new Error(`Saga not found: ${sagaId}`);
    }

    const referenceTime = saga.lastSummarizedAt ?? new Date(0);
    // retrieveEpisodes returns newest-first; the LLM summary expects narrative
    // order (oldest-first) so events read sequentially.
    const newEpisodes = (
      await this.episodicNodeRepository.retrieveEpisodes(
        RetrieveEpisodesParamsSchema.parse({
          referenceTime: new Date(),
          lastN: 100,
          graphIds: [graphId],
          sagaId,
        }),
      )
    ).reverse();

    const unsummarized = newEpisodes.filter((ep) => ep.validAt > referenceTime);

    if (unsummarized.length === 0) {
      return {
        summary: saga.summary,
        metrics: { ...baseMetrics, 'episodes.unsummarized': 0 },
      };
    }

    const messages = buildSummarizeSagasMessages({
      sagaName: saga.name,
      existingSummary: saga.summary,
      newEpisodes: unsummarized,
    });
    const result = await invokeStructured(model, SagaSummarySchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'summarize-saga',
      tags: ['knowledge-graph', 'saga.summary'],
    });

    const updatedSaga = {
      ...saga,
      summary: result.summary,
      lastSummarizedAt: new Date(),
    };
    await this.sagaRepository.save(updatedSaga);

    return {
      summary: updatedSaga.summary,
      metrics: { ...baseMetrics, 'episodes.unsummarized': unsummarized.length },
    };
  }

  async addTextEpisodes(
    options: AddTextEpisodesOptionsInput,
  ): Promise<AddEpisodeResult[]> {
    const parsed = AddTextEpisodesOptionsSchema.parse(options);
    const normalized: NormalizedAddEpisodeOptions = {
      ...parsed,
      episodes: parsed.episodes.map((ep) => ({
        ...ep,
        source: EpisodeType.text,
        referenceTime: new Date(ep.referenceTime),
      })),
    };
    const { results } = await this.addEpisodesImpl(
      normalized,
      EpisodeService.makeCtx(normalized),
    );
    return results;
  }

  async addMessageEpisodes(
    options: AddMessageEpisodesOptionsInput,
  ): Promise<AddEpisodeResult[]> {
    const parsed = AddMessageEpisodesOptionsSchema.parse(options);
    const normalized: NormalizedAddEpisodeOptions = {
      ...parsed,
      episodes: parsed.episodes.map((ep) => ({
        ...ep,
        source: EpisodeType.message,
        referenceTime: new Date(ep.referenceTime),
        content: ep.content.map((t) => `${t.speakerName}: ${t.message}`).join('\n'),
      })),
    };
    const { results } = await this.addEpisodesImpl(
      normalized,
      EpisodeService.makeCtx(normalized),
    );
    return results;
  }

  async addJsonEpisodes(
    options: AddJsonEpisodesOptionsInput,
  ): Promise<AddEpisodeResult[]> {
    const parsed = AddJsonEpisodesOptionsSchema.parse(options);
    const normalized: NormalizedAddEpisodeOptions = {
      ...parsed,
      episodes: parsed.episodes.map((ep) => ({
        ...ep,
        source: EpisodeType.json,
        referenceTime: new Date(ep.referenceTime),
      })),
    };
    const { results } = await this.addEpisodesImpl(
      normalized,
      EpisodeService.makeCtx(normalized),
    );
    return results;
  }

  /**
   * Batch ingestion orchestrator. Runs the pipeline as a sequence of phases that
   * thread a per-episode `EpisodeWorkItem[]` plus a graph-global `BatchState`,
   * then assembles one `AddEpisodeResult` per input episode.
   *
   * Phases fan out with plain `Promise.all`: LLM generations are already
   * bounded by the distributed semaphore (README, "Concurrency & rate limiting").
   */
  @Span('addEpisodes', { onResult: metricsOnResult })
  private async addEpisodesImpl(
    parsed: NormalizedAddEpisodeOptions,
    ctx: LlmContext,
  ): Promise<{ results: AddEpisodeResult[]; metrics: SpanMetrics }> {
    const startMs = performance.now();
    const { userId, episodes, updateCommunities, resolveCoreferences } = parsed;

    const cfg: PipelineConfig = {
      entityTypes: parsed.entityTypes,
      edgeTypes: parsed.edgeTypes,
      effectiveEdgeTypeMappings: getEffectiveTypeMappings(
        parsed.edgeTypeMappings,
        parsed.edgeTypes,
      ),
      excludedEntityTypes: parsed.excludedEntityTypes,
      customInstructions: parsed.customInstructions,
      resolveCoreferences,
    };
    // TODO: add abort on job cancellation when moving to ingestion within a job
    // See community service for implementation
    const abort = new AbortController();

    // In-flight requests each attach a signal listener; the pool size can exceed
    // Node's default 10-listener warning threshold.
    setMaxListeners(0, abort.signal);
    const model = await this.llmService.getActiveModel(userId, {
      abortSignal: abort.signal,
    });

    const { items, batch } = await this.preparePhase(parsed);
    try {
      await this.nodesPhase(items, batch, model, cfg, ctx);
      await this.edgesPhase(items, batch, model, cfg, ctx);
      await this.enrichPhase(items, batch, model, cfg, ctx);
    } catch (e) {
      abort.abort();
      throw e;
    }
    await this.persistPhase(items, batch);

    // The maintenance service routes each distinct graph to a
    // debounced full rebuild or the incremental update path based on its size.
    const graphIds = [...new Set(items.map((it) => it.episode.graphId))];
    if (updateCommunities) {
      for (const gid of graphIds) {
        const entityIds = batch.canonicalNodes
          .filter((n) => n.graphId === gid)
          .map((n) => n.id);
        if (entityIds.length === 0) continue;

        await this.communityMaintenance.scheduleMaintenance(userId, gid, entityIds);
      }
    }

    // TODO: per-entry `nodes` includes both newly-resolved canonical nodes AND
    // preexisting nodes matched via cross-batch dedup. The same canonical EntityNode
    // may therefore appear in multiple entries' `nodes` arrays - callers must
    // dedupe by id if they want a unique set across the batch.
    const results = items.map((it): AddEpisodeResult => {
      // A reference an edge extraction claimed by id is no longer unresolved.
      const claimedRefIds = new Set(
        it.committedCorefBindings
          .map((b) => b.resolvedUnresolvedReferenceId)
          .filter((id): id is Uuid => id !== null),
      );
      return {
        episode: it.episode,
        nodes: it.canonicalNodes,
        edges: it.edgeResolution.resolvedEdges,
        invalidatedEdges: it.edgeResolution.invalidatedEdges,
        episodicEdges: it.episodicEdges,
        unresolvedReferences: it.unresolvedReferences
          .filter((r) => !claimedRefIds.has(r.id))
          .map(({ surfaceForm, locatingQuote }) => ({ surfaceForm, locatingQuote })),
      };
    });

    return {
      results,
      metrics: {
        'user.id': ctx.userId,
        'session.id': ctx.sessionId,
        'episode.count': episodes.length,
        'episode.ids': items.map((it) => it.episode.id).join(','),
        'graph.ids': graphIds.join(','),
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.new': batch.canonicalNodes.filter(
          (n) => !batch.preexistingGraphNodeIds.has(n.id),
        ).length,
        'edge.count.resolved': items.reduce(
          (s, it) => s + it.edgeResolution.resolvedEdges.length,
          0,
        ),
        'edge.count.invalidated': items.reduce(
          (s, it) => s + it.edgeResolution.invalidatedEdges.length,
          0,
        ),
        'edge.count.new': items.reduce(
          (s, it) => s + it.edgeResolution.newEdges.length,
          0,
        ),
        'previousEpisodes.totalCount': items.reduce(
          (s, it) => s + it.prevEpisodes.length,
          0,
        ),
        updateCommunities: updateCommunities,
        duration_ms: Math.round(performance.now() - startMs),
      },
    };
  }

  /**
   * Phase 1 - prepare. Retrieves previous-episode context, creates the episodic
   * nodes, chunks each episode once, and assembles the `EpisodeWorkItem[]` and an
   * empty `BatchState`. Collection fields start empty so each item is fully typed
   * from construction.
   */
  @Span('preparePhase', { onResult: metricsOnResult })
  private async preparePhase(
    parsed: NormalizedAddEpisodeOptions,
  ): Promise<{ items: EpisodeWorkItem[]; batch: BatchState; metrics: SpanMetrics }> {
    const { episodes } = parsed;

    // TODO: upstream's singular `add_episode` filters previous episodes by
    // source (graphiti.py:1045 - `source=source`). Upstream's bulk path doesn't.
    // We took the bulk semantics; revisit if same-source context proves to
    // matter for extraction quality.
    const prevEpisodesPerEpisode = await Promise.all(
      episodes.map((ep) =>
        this.episodicNodeRepository.retrieveEpisodes(
          RetrieveEpisodesParamsSchema.parse({
            referenceTime: ep.referenceTime,
            lastN: PREVIOUS_EPISODES_WINDOW,
            graphIds: [ep.graphId],
          }),
        ),
      ),
    );

    const items: EpisodeWorkItem[] = episodes.map((raw, i) => {
      const base = createEpisodicNode({
        name: raw.name,
        content: raw.content,
        source: raw.source,
        sourceDescription: raw.sourceDescription,
        graphId: raw.graphId,
        validAt: raw.referenceTime,
        sagaId: raw.sagaId ?? null,
      });
      const episode = raw.id ? { ...base, id: raw.id } : base;

      return {
        episode,
        chunks: prepareChunks(episode.content, episode.source),
        prevEpisodes: prevEpisodesPerEpisode[i],

        chunkIndicesByExtractedId: new Map(),
        corefByExtractedId: new Map(),
        unresolvedReferences: [],
        nodeResolution: {
          newNodes: [],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
        },
        canonicalNodes: [],

        committedCorefBindings: [],
        selfLoopFactsForEnrichment: [],
        edgeResolution: { resolvedEdges: [], invalidatedEdges: [], newEdges: [] },

        episodicEdges: [],
      };
    });

    const batch: BatchState = {
      canonicalIdByNodeId: new Map(),
      allKnownNodesById: new Map(),
      preexistingGraphNodeIds: new Set(),
      chunkSources: new Map(),
      canonicalNodes: [],
      corefByCanonicalId: new Map(),
    };

    return {
      items,
      batch,
      metrics: {
        'episode.count': episodes.length,
        'previousEpisodes.totalCount': prevEpisodesPerEpisode.reduce(
          (s, a) => s + a.length,
          0,
        ),
      },
    };
  }

  /**
   * Phase 2 - nodes. Extracts and embeds entity nodes per episode, resolves them
   * against the live graph (pass 1) and within the batch (pass 2), and writes the
   * canonical id map, known-node map and per-item canonical node sets.
   */
  @Span('nodesPhase', { onResult: metricsOnResult })
  private async nodesPhase(
    items: EpisodeWorkItem[],
    batch: BatchState,
    model: BaseChatModel,
    cfg: PipelineConfig,
    ctx: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    // Extract nodes per episode
    const nodeExtractions = await withConcurrency(
      this.kgConfig.memoryBackpressureConcurrencyLimit,
      items.map(
        (it) => () =>
          this.nodeExtractionService.extractNodes(
            model,
            it.episode,
            it.chunks,
            it.prevEpisodes,
            cfg.entityTypes,
            cfg.customInstructions,
            cfg.excludedEntityTypes,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.episode.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.chunkIndicesByExtractedId = nodeExtractions[i].chunkIndicesByExtractedId;
      it.corefByExtractedId = nodeExtractions[i].corefByExtractedId;
      it.unresolvedReferences = nodeExtractions[i].unresolvedReferences;
    });

    // Embed all extracted nodes in one batched call, scatter back per episode.
    const allExtractedNodes = nodeExtractions.flatMap((r) => r.nodes);
    const allEmbedded = await this.embeddingService.embedNodes(allExtractedNodes);
    const extractedNodesPerItem = reassembleByOffsets(
      allEmbedded,
      nodeExtractions.map((r) => r.nodes.length),
    );

    // Pass 1 - resolve vs live graph (parallel). resolveNodes collects its own
    // candidates; seed allKnownNodesById + the preexisting-id set so a new node can collapse
    // onto a preexisting one even when it wasn't in its candidate set.
    const resolutions = await withConcurrency(
      this.kgConfig.memoryBackpressureConcurrencyLimit,
      items.map(
        (it, i) => () =>
          this.nodeResolutionService.resolveNodes(
            model,
            it.episode,
            it.chunks,
            it.chunkIndicesByExtractedId,
            extractedNodesPerItem[i],
            it.prevEpisodes,
            cfg.customInstructions,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.episode.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.nodeResolution = resolutions[i];
    });

    for (const it of items) {
      for (const cand of it.nodeResolution.preexistingCandidates) {
        batch.allKnownNodesById.set(cand.id, cand);
        batch.preexistingGraphNodeIds.add(cand.id);
      }
    }

    // Pass 2 - within-batch dedup over all new nodes seeded with matched-preexisting.
    const preexistingMatchPairs: [aliasId: Uuid, canonicalId: Uuid][] = items.flatMap(
      (it) =>
        it.nodeResolution.nodesMatchedToPreexistingNodes.map(
          (p): [aliasId: Uuid, canonicalId: Uuid] => [p.extractedId, p.preexistingNodeId],
        ),
    );
    const allNewNodes = items.flatMap((it) => it.nodeResolution.newNodes);
    const matchedPreexistingIds = new Set(
      items.flatMap((it) =>
        it.nodeResolution.nodesMatchedToPreexistingNodes.map((p) => p.preexistingNodeId),
      ),
    );
    const matchedPreexistingNodes = [...matchedPreexistingIds]
      .map((id) => batch.allKnownNodesById.get(id))
      .filter((n): n is EntityNode => n !== undefined);

    const withinBatchDedupePairs = this.nodeResolutionService.dedupeAcrossBatch(
      allNewNodes,
      matchedPreexistingNodes,
    );

    batch.canonicalIdByNodeId = buildDirectedIdMap([
      ...preexistingMatchPairs,
      ...withinBatchDedupePairs,
    ]);
    for (const n of allNewNodes) batch.allKnownNodesById.set(n.id, n);

    recomputeCorefByCanonicalIdMap(items, batch);
    recomputeCanonicalNodesByCanonicalIdMap(items, batch);

    return {
      metrics: {
        'node.count.extracted': allExtractedNodes.length,
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.new': batch.canonicalNodes.filter(
          (n) => !batch.preexistingGraphNodeIds.has(n.id),
        ).length,
      },
    };
  }

  /**
   * Folds within-episode node canonicalization pairs into the batch id map and
   * downstream state. Example: extraction produced both a "captain" node and
   * a "Captain Clark" node from different chunks, and the canonicalization LLM decided
   * they are the same person - the pair is ("captain".id, "Captain Clark".id)
   *
   * - both new: merge as proposed;
   * - mergedAway new, kept preexisting: merge as proposed (the phantom's chunks,
   *   edges and descriptor route to the live node);
   * - mergedAway preexisting, kept new: FLIP - if "captain" is already a database
   *   row and "Captain Clark" was extracted fresh, keep the live "captain" id
   *   (its persisted edges stay valid) but rename that node to "Captain Clark"
   *   (re-embedded by enrichPhase's renamed-nodes pass);
   * - both preexisting (incl. a second live row flipping onto the same kept node):
   *   drop - a true live-node merge repoints persisted edges and deletes a row,
   *   which is cross-episode consolidation, not batch state.
   *   TODO: both preexisting means one entity is duplicated,
   *   which must be addressed somehow
   *
   * Reads:
   *   batch.preexistingGraphNodeIds
   *   batch.allKnownNodesById
   * Writes:
   *   batch.canonicalIdByNodeId (rebuilt with the accepted pairs)
   *   flipped live nodes' name + nameEmbedding (mutated in place)
   *   items[].committedCorefBindings (boundNodeId remapped)
   *   batch.corefByCanonicalId (via recomputeCorefByCanonicalId)
   *   items[].canonicalNodes (via recomputeCanonicalNodes)
   *   batch.canonicalNodes (via recomputeCanonicalNodes)
   */
  private mergeNodeCanonicalizationPairs(
    proposed: [mergedAwayId: Uuid, keptId: Uuid][],
    items: EpisodeWorkItem[],
    batch: BatchState,
  ): { applied: number; flipped: number; dropped: string[] } {
    const pairs: [aliasId: Uuid, canonicalId: Uuid][] = [];
    const dropped: string[] = [];
    const flipTargetByKept = new Map<Uuid, Uuid>();

    for (const [mergedAway, kept] of proposed) {
      const mergedAwayPreexisting = batch.preexistingGraphNodeIds.has(mergedAway);
      const keptPreexisting = batch.preexistingGraphNodeIds.has(kept);

      if (mergedAwayPreexisting && (keptPreexisting || flipTargetByKept.has(kept))) {
        dropped.push(`${mergedAway}->${kept}`);
        continue;
      }

      if (mergedAwayPreexisting) {
        const liveNode = batch.allKnownNodesById.get(mergedAway);
        const newNode = batch.allKnownNodesById.get(kept);
        if (!liveNode || !newNode) {
          throw new Error(
            `canonicalization fold: pair ${mergedAway} -> ${kept} references a node unknown to the batch`,
          );
        }
        if (liveNode.name !== newNode.name) {
          liveNode.name = newNode.name;
          liveNode.nameEmbedding = null;
        }
        pairs.push([kept, mergedAway]);
        flipTargetByKept.set(kept, mergedAway);
        continue;
      }
      pairs.push([mergedAway, kept]);
    }

    if (pairs.length > 0) {
      batch.canonicalIdByNodeId = buildDirectedIdMap([
        ...batch.canonicalIdByNodeId.entries(),
        ...pairs,
      ]);
      // TODO: No post-fold readers of corefByCanonicalId yet; kept consistent for deferred coref work.
      recomputeCorefByCanonicalIdMap(items, batch);
      recomputeCanonicalNodesByCanonicalIdMap(items, batch);

      items.forEach((it) => {
        it.committedCorefBindings = it.committedCorefBindings.map((b) => ({
          ...b,
          boundNodeId: batch.canonicalIdByNodeId.get(b.boundNodeId) ?? b.boundNodeId,
        }));
      });
    }
    return { applied: pairs.length, flipped: flipTargetByKept.size, dropped };
  }

  /**
   * Phase 3 - edges. Extracts and embeds edges per episode against the canonical
   * nodes, dedupes across the batch, routes each canonical edge back to its origin
   * episode for resolution, then fills attributes / timestamp fallbacks on the
   * freshly extracted (new) edges only.
   */
  @Span('edgesPhase', { onResult: metricsOnResult })
  private async edgesPhase(
    items: EpisodeWorkItem[],
    batch: BatchState,
    model: BaseChatModel,
    cfg: PipelineConfig,
    ctx: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    // Extract edges per episode using this episode's canonical nodes, while a concurrent
    // per-episode canonicalization pass groups duplicate nodes from different chunks
    // Extraction sees the pre-fold node list and the endpoint remap below absorbs the difference.
    const sharedSemaphore = new CountingSemaphore(
      this.kgConfig.memoryBackpressureConcurrencyLimit,
    );
    const [edgeExtractions, canonPairsPerItem] = await Promise.all([
      withConcurrency(
        sharedSemaphore,
        items.map((it) => () => {
          return this.edgeExtractionService.extractEdges(
            model,
            it.episode,
            it.chunks,
            it.canonicalNodes,
            it.prevEpisodes,
            cfg.customInstructions,
            cfg.edgeTypes,
            cfg.effectiveEdgeTypeMappings,
            cfg.resolveCoreferences ?? false,
            cfg.resolveCoreferences ? buildEdgeCorefCandidates(it, batch) : [],
            cfg.resolveCoreferences ? it.unresolvedReferences : [],
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.episode.id } },
          );
        }),
      ),
      withConcurrency(
        sharedSemaphore,
        items.map(
          (it) => (): Promise<[mergedAwayId: Uuid, keptId: Uuid][]> =>
            it.chunks.length > 1
              ? this.nodeResolutionService.canonicalizeEpisodeNodes(
                  model,
                  it.canonicalNodes,
                  batch.corefByCanonicalId,
                  { ...ctx, metadata: { ...ctx.metadata, episodeId: it.episode.id } },
                )
              : Promise.resolve([]),
        ),
      ),
    ]);
    items.forEach((it, i) => {
      it.committedCorefBindings = edgeExtractions[i].committedCorefBindings;
    });

    const canonicalization = this.mergeNodeCanonicalizationPairs(
      canonPairsPerItem.flat(),
      items,
      batch,
    );

    // Edge chunk provenance keyed by edge id. INVARIANT: episodeIndex == the
    // item's position; every [][] view handed to the resolution service below is
    // derived from `items` in order so the indices stay aligned.
    edgeExtractions.forEach((extraction, i) => {
      for (const [id, indices] of extraction.chunkIndicesByEdgeId) {
        batch.chunkSources.set(id, { episodeIndex: i, indices });
      }
    });

    // Remap endpoints through node dedup (incl. the canonicalization fold),
    // then pull self-loops - both deliberate single-entity facts from
    // extraction and pairs whose mentions collapsed onto one entity here.
    // They feed node enrichment as context (enrichPhase) instead of persisting
    // as self-loop edges. Then embed the surviving edges in one call.
    const rawEdgesPerItem = items.map((it, i) => {
      const remapped = remapEdgeEndpointsToCanonical(
        edgeExtractions[i].edges,
        batch.canonicalIdByNodeId,
      );
      it.selfLoopFactsForEnrichment = remapped.filter(
        (e) => e.sourceNodeId === e.targetNodeId,
      );
      return remapped.filter((e) => e.sourceNodeId !== e.targetNodeId);
    });
    const allExtractedEdges = rawEdgesPerItem.flat();

    const allEmbeddedEdges = await this.embeddingService.embedEdges(allExtractedEdges);
    const embeddedEdgesPerItem = reassembleByOffsets(
      allEmbeddedEdges,
      rawEdgesPerItem.map((edges) => edges.length),
    );

    // Cross-batch edge dedup -> flat canonical set. Mirrors `dedupe_edges_bulk`.
    const canonicalEdges = await this.edgeResolutionService.dedupeAcrossBatch(
      model,
      embeddedEdgesPerItem,
      items.map((it) => it.episode),
      items.map((it) => it.chunks),
      batch.chunkSources,
      items.map((it) => it.prevEpisodes),
      cfg.customInstructions,
      ctx,
    );

    // Route each canonical edge to its ORIGIN episode so it is deduped exactly
    // once, against the episode whose previousEpisodes / chunks made it.
    const edgesByOriginItem: EntityEdge[][] = items.map(() => []);
    for (const edge of canonicalEdges) {
      const source = batch.chunkSources.get(edge.id);
      if (!source) {
        throw new Error(`edge dedup partition: edge ${edge.id} has no chunk source`);
      }
      edgesByOriginItem[source.episodeIndex].push(edge);
    }

    // DEDUPE per origin episode (candidates collected inside dedupeEdges). No
    // timestamps are touched here - contradictions are recorded and resolved
    // after enrichment fills validAt/invalidAt.
    const dedupes = await withConcurrency(
      this.kgConfig.memoryBackpressureConcurrencyLimit,
      items.map(
        (it, i) => () =>
          this.edgeResolutionService.dedupeEdges(
            model,
            items.map((other) => other.episode),
            items.map((other) => other.chunks),
            batch.chunkSources,
            edgesByOriginItem[i],
            batch.canonicalIdByNodeId,
            it.episode.validAt,
            it.prevEpisodes,
            cfg.customInstructions,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.episode.id } },
          ),
      ),
    );

    // FILL - one chunk-grounded enrichment call per new edge (timestamps +
    // custom attributes), over the pooled new edges. Mutates edges in place.
    const allNewEdges = dedupes.flatMap((d) => d.newEdges);
    await this.edgeExtractionService.enrichEdges(
      model,
      allNewEdges,
      batch.canonicalNodes,
      items.map((it) => it.episode),
      items.map((it) => it.chunks),
      batch.chunkSources,
      cfg.edgeTypes,
      cfg.effectiveEdgeTypeMappings,
      items.map((it) => it.committedCorefBindings),
      ctx,
    );

    // INVALIDATE - temporal adjudication over the now-filled new edges, global so
    // a preexisting edge contradicted from two episodes is invalidated once.
    //
    // TODO: only new-vs-preexisting-graph contradictions are adjudicated. Two
    // brand-new edges from different episodes in this batch that contradict each
    // other are never compared (dedupeAcrossBatch collapses duplicates but
    // ignores contradictions), so both persist. Examine with cross-graph
    // ingestion in mind - a batch may span multiple graphs, so any new-vs-new
    // pass must scope contradictions by graphId (never across graphs).
    const mergedContradictions = new Map<Uuid, EntityEdge[]>();
    for (const dedupe of dedupes) {
      for (const [id, edges] of dedupe.contradictionsByNewEdgeId) {
        mergedContradictions.set(id, edges);
      }
    }
    const { invalidatedByNewEdgeId } = this.edgeResolutionService.invalidateEdges(
      allNewEdges,
      mergedContradictions,
    );

    // Reassemble the per-item EdgeResolutionResult. Each new edge belongs to one
    // origin episode (chunkSources), so invalidations attribute to that entry.
    items.forEach((it, i) => {
      const dedupe = dedupes[i];
      it.edgeResolution = {
        newEdges: dedupe.newEdges,
        resolvedEdges: [...dedupe.matchedPreexistingEdges, ...dedupe.newEdges],
        invalidatedEdges: dedupe.newEdges.flatMap(
          (s) => invalidatedByNewEdgeId.get(s.id) ?? [],
        ),
      };
    });

    return {
      metrics: {
        'edge.count.extracted': allExtractedEdges.length,
        'edge.count.resolved': items.reduce(
          (s, it) => s + it.edgeResolution.resolvedEdges.length,
          0,
        ),
        'edge.count.invalidated': items.reduce(
          (s, it) => s + it.edgeResolution.invalidatedEdges.length,
          0,
        ),
        'edge.count.new': allNewEdges.length,
        'canonicalization.pairs.applied': canonicalization.applied,
        'canonicalization.pairs.flipped': canonicalization.flipped,
        // Live-live merges deferred to cross-episode consolidation; ids kept
        // visible so their frequency can be judged against a real graph.
        'canonicalization.pairs.dropped': canonicalization.dropped.join(','),
        'edge.count.selfLoopsFolded': items.reduce(
          (s, it) => s + it.selfLoopFactsForEnrichment.length,
          0,
        ),
      },
    };
  }

  /**
   * Phase 4 - enrich. Fills entity attributes and summaries on the canonical
   * nodes and re-embeds nodes renamed during dedup. All DB writes happen in
   * persistPhase.
   */
  @Span('enrichPhase', { onResult: metricsOnResult })
  private async enrichPhase(
    items: EpisodeWorkItem[],
    batch: BatchState,
    model: BaseChatModel,
    cfg: PipelineConfig,
    ctx: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const allResolvedEdges = items.flatMap((it) => it.edgeResolution.resolvedEdges);
    const allNewEdges = items.flatMap((it) => it.edgeResolution.newEdges);
    // Self-loops folded at the canonicalization remap: single-entity facts that
    // feed enrichment context only (never deduped, filled, or persisted).
    const foldedFacts = items.flatMap((it) => it.selfLoopFactsForEnrichment);

    const nodeContext = buildNodeContext(
      items.map((it) => it.canonicalNodes),
      items.map((it) => it.chunkIndicesByExtractedId),
      batch.canonicalIdByNodeId,
      items.map((it) => it.episode),
      items.map((it) => it.prevEpisodes),
      items.map((it) => it.chunks),
      items.map((it) => it.committedCorefBindings),
    );

    // Entity attributes refined from this episode's content, with resolved-edge
    // context. Runs over the full resolved set (new + matched preexisting).
    await this.nodeExtractionService.fillEntityAttributes(
      model,
      batch.canonicalNodes,
      [...allResolvedEdges, ...foldedFacts],
      cfg.entityTypes,
      nodeContext,
      ctx,
    );

    // Summaries for all canonical nodes; only NEW edges as fact context so
    // matched-preexisting edges aren't re-emitted as known facts.
    await this.nodeExtractionService.summarizeNodes(
      model,
      batch.canonicalNodes,
      [...allNewEdges, ...foldedFacts],
      cfg.entityTypes,
      nodeContext,
      ctx,
    );

    // Re-embed nodes renamed during dedup (resolution rewrites name + nulls the
    // stale nameEmbedding). Write fresh objects back by id into BOTH the batch
    // set and each item's canonicalNodes so persistence and the result agree.
    const renamedNodes = batch.canonicalNodes.filter((n) => n.nameEmbedding === null);
    if (renamedNodes.length > 0) {
      const reEmbedded = await this.embeddingService.embedNodes(renamedNodes);
      const byId = new Map(reEmbedded.map((n) => [n.id, n]));
      const replace = (nodes: EntityNode[]) => nodes.map((n) => byId.get(n.id) ?? n);

      batch.canonicalNodes = replace(batch.canonicalNodes);
      items.forEach((it) => {
        it.canonicalNodes = replace(it.canonicalNodes);
      });
    }

    return {
      metrics: {
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.reEmbedded': renamedNodes.length,
        'edge.count.new': allNewEdges.length,
      },
    };
  }

  /**
   * Phase 5 - persist. Builds the MENTIONS edges and saga nodes, then writes
   * everything in FK-correct order in a single transaction: sagas first
   * (episodic nodes FK onto them via saga_id), then entity + episodic nodes,
   * then entity / episodic edges.
   */
  @Span('persistPhase', { onResult: metricsOnResult })
  private async persistPhase(
    items: EpisodeWorkItem[],
    batch: BatchState,
  ): Promise<{ metrics: SpanMetrics }> {
    items.forEach((it) => {
      it.episodicEdges = it.canonicalNodes.map((node) =>
        createEpisodicEdge({
          sourceNodeId: it.episode.id,
          targetNodeId: node.id,
          graphId: it.episode.graphId,
        }),
      );
    });

    // One Saga per distinct sagaId
    // TODO: saga name defaults to the ID string. Plan: accept an optional
    // caller-provided name, otherwise let summarizeSaga generate one.
    const graphIdBySagaId = new Map<Uuid, Uuid>();
    for (const it of items) {
      const sagaId = it.episode.sagaId;
      if (sagaId && !graphIdBySagaId.has(sagaId)) {
        graphIdBySagaId.set(sagaId, it.episode.graphId);
      }
    }
    const sagas = [...graphIdBySagaId].map(([sagaId, graphId]) =>
      createSaga({ id: sagaId, name: NodeNameSchema.parse(sagaId), graphId }),
    );
    const allResolvedEdges = items.flatMap((it) => it.edgeResolution.resolvedEdges);
    const allInvalidatedEdges = items.flatMap((it) => it.edgeResolution.invalidatedEdges);
    const allEpisodicEdges = items.flatMap((it) => it.episodicEdges);

    // in FK order: sagas first (episodic nodes FK onto them),
    // then entity + episodic nodes, then entity / episodic edges.
    await this.prisma.$transaction(
      async (tx) => {
        for (const saga of sagas) {
          await this.sagaRepository.createIfNotExists(saga, tx);
        }
        await this.entityNodeRepository.saveBulk(batch.canonicalNodes, tx);
        await this.episodicNodeRepository.saveBulk(
          items.map((it) => it.episode),
          tx,
        );
        await this.entityEdgeRepository.saveBulk(allResolvedEdges, tx);
        await this.entityEdgeRepository.saveBulk(allInvalidatedEdges, tx);
        await this.episodicEdgeRepository.saveBulk(allEpisodicEdges, tx);
      },
      {
        timeout: PERSIST_TRANSACTION_TIMEOUT_MS,
        maxWait: PERSIST_TRANSACTION_MAX_WAIT_MS,
      },
    );

    return {
      metrics: {
        'node.count.persisted': batch.canonicalNodes.length,
        'episode.count': items.length,
        'saga.count': sagas.length,
        'edge.count.resolved': allResolvedEdges.length,
        'edge.count.invalidated': allInvalidatedEdges.length,
      },
    };
  }
}
