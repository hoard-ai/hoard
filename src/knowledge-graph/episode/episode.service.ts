import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

import { Uuid } from '@/common/schemas';
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

import {
  buildDirectedIdMap,
  LLM_CONCURRENCY_LIMIT,
  reassembleByOffsets,
  resolveEdgePointers,
  withConcurrency,
} from '../batch-utils';
import { CommunityMaintenanceService } from '../community';
import { EmbeddingService } from '../embedding';
import { EdgeExtractionService, NodeExtractionService } from '../extraction';
import {
  createEpisodicEdge,
  createEpisodicNode,
  createHasEpisodeEdge,
  createSagaNode,
  EntityNode,
  EpisodicNode,
  HasEpisodeEdge,
} from '../models';
import { buildSummarizeSagasMessages, SagaSummarySchema } from '../prompts';
import {
  EntityEdgeRepository,
  EntityNodeRepository,
  EpisodicEdgeRepository,
  EpisodicNodeRepository,
  HasEpisodeEdgeRepository,
  SagaNodeRepository,
} from '../repository';
import { EdgeResolutionService, NodeResolutionService } from '../resolution';
import {
  EpisodeType,
  NodeNameSchema,
  RetrieveEpisodesParamsInput,
  RetrieveEpisodesParamsSchema,
} from '../types';
import { prepareChunks } from './content-chunking';
import { buildNodeContext, getEffectiveTypeMappings } from './episode-utils';
import {
  AddEpisodeResult,
  AddJsonEpisodesOptionsInput,
  AddJsonEpisodesOptionsSchema,
  AddMessageEpisodesOptionsInput,
  AddMessageEpisodesOptionsSchema,
  AddTextEpisodesOptionsInput,
  AddTextEpisodesOptionsSchema,
  BatchState,
  EpisodeWorkItem,
  NormalizedAddEpisodeOptions,
  PipelineConfig,
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
    private readonly sagaNodeRepository: SagaNodeRepository,
    private readonly hasEpisodeEdgeRepository: HasEpisodeEdgeRepository,
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
  async deleteEpisode(id: Uuid): Promise<void> {
    await this.deleteEpisodeImpl(id);
  }

  @Span('deleteEpisode', { onResult: metricsOnResult })
  private async deleteEpisodeImpl(id: Uuid): Promise<{ metrics: SpanMetrics }> {
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
  async deleteEpisodesById(ids: Uuid[]): Promise<void> {
    await this.deleteEpisodesByIdImpl(ids);
  }

  @Span('deleteEpisodesById', { onResult: metricsOnResult })
  private async deleteEpisodesByIdImpl(ids: Uuid[]): Promise<{ metrics: SpanMetrics }> {
    await Promise.all(ids.map((id) => this.deleteEpisode(id)));
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

    const saga = await this.sagaNodeRepository.getById(sagaId);
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
    await this.sagaNodeRepository.save(updatedSaga);

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
   * then assembles one `AddEpisodeResult` per input episode. Each phase carries
   * its own span and structural-count metrics; this root span keeps the summary
   * rollup, read off the threaded state rather than recomputed.
   */
  @Span('addEpisodes', { onResult: metricsOnResult })
  private async addEpisodesImpl(
    parsed: NormalizedAddEpisodeOptions,
    ctx: LlmContext,
  ): Promise<{ results: AddEpisodeResult[]; metrics: SpanMetrics }> {
    const startMs = performance.now();
    const { userId, episodes, updateCommunities } = parsed;

    const cfg: PipelineConfig = {
      entityTypes: parsed.entityTypes,
      edgeTypes: parsed.edgeTypes,
      effectiveEdgeTypeMappings: getEffectiveTypeMappings(
        parsed.edgeTypeMappings,
        parsed.edgeTypes,
      ),
      excludedEntityTypes: parsed.excludedEntityTypes,
      customInstructions: parsed.customInstructions,
      updateCommunities,
    };
    const model = await this.llmService.getActiveModel(userId);

    const { items, batch } = await this.preparePhase(parsed);
    await this.nodesPhase(items, batch, model, cfg, ctx);
    await this.edgesPhase(items, batch, model, cfg, ctx);
    await this.enrichPhase(items, batch, model, cfg, ctx);
    await this.persistPhase(items, batch);

    // The maintenance service routes each distinct graph to a
    // debounced full rebuild or the incremental update path based on its size.
    const graphIds = [...new Set(items.map((it) => it.node.graphId))];
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
    // existing nodes matched via cross-batch dedup. The same canonical EntityNode
    // may therefore appear in multiple entries' `nodes` arrays - callers must
    // dedupe by id if they want a unique set across the batch.
    const results = items.map(
      (it): AddEpisodeResult => ({
        episode: it.node,
        nodes: it.canonicalNodes,
        edges: it.edgeResolution.resolvedEdges,
        invalidatedEdges: it.edgeResolution.invalidatedEdges,
        episodicEdges: it.episodicEdges,
      }),
    );

    return {
      results,
      metrics: {
        'user.id': ctx.userId,
        'session.id': ctx.sessionId,
        'episode.count': episodes.length,
        'episode.ids': items.map((it) => it.node.id).join(','),
        'graph.ids': graphIds.join(','),
        'node.count.extracted': items.reduce((s, it) => s + it.extractedNodes.length, 0),
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.new': batch.canonicalNodes.filter(
          (n) => !batch.existingNodeIds.has(n.id),
        ).length,
        'edge.count.extracted': items.reduce((s, it) => s + it.rawEdges.length, 0),
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
      });
      const node = raw.id ? { ...base, id: raw.id } : base;

      return {
        node,
        chunks: prepareChunks(node.content, node.source),
        prevEpisodes: prevEpisodesPerEpisode[i],
        sagaId: raw.sagaId,
        extractedNodes: [],
        chunkIndicesByNodeId: new Map(),
        resolution: {
          resolvedNodes: [],
          idMap: new Map(),
          duplicatePairs: [],
          candidates: [],
        },
        canonicalNodes: [],
        rawEdges: [],
        chunkIndicesByEdgeId: new Map(),
        edgesFromThisEpisode: [],
        edgeResolution: { resolvedEdges: [], invalidatedEdges: [], newEdges: [] },
        episodicEdges: [],
      };
    });

    const batch: BatchState = {
      canonicalIdByNodeId: new Map(),
      nodeRegistry: new Map(),
      existingNodeIds: new Set(),
      chunkSources: new Map(),
      canonicalNodes: [],
      sagaNodes: [],
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
   * canonical id map, node registry and per-item canonical node sets.
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
      LLM_CONCURRENCY_LIMIT,
      items.map(
        (it) => () =>
          this.nodeExtractionService.extractNodes(
            model,
            it.node,
            it.chunks,
            it.prevEpisodes,
            cfg.entityTypes,
            cfg.customInstructions,
            cfg.excludedEntityTypes,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.node.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.extractedNodes = nodeExtractions[i].nodes;
      it.chunkIndicesByNodeId = nodeExtractions[i].chunkIndicesByNodeId;
    });

    // Embed all extracted nodes in one batched call, scatter back onto items.
    const allExtractedNodes = items.flatMap((it) => it.extractedNodes);
    const allEmbedded = await this.embeddingService.embedNodes(allExtractedNodes);
    const embeddedPerItem = reassembleByOffsets(
      allEmbedded,
      items.map((it) => it.extractedNodes.length),
    );
    items.forEach((it, i) => {
      it.extractedNodes = embeddedPerItem[i];
    });

    // Pass 1 - resolve vs live graph (parallel). resolveNodes collects its own
    // candidates; seed the registry + existing-id set so a new node can collapse
    // onto an existing one even when that existing one wasn't in its candidate set.
    const resolutions = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      items.map(
        (it) => () =>
          this.nodeResolutionService.resolveNodes(
            model,
            it.node,
            it.chunks,
            it.chunkIndicesByNodeId,
            it.extractedNodes,
            it.prevEpisodes,
            cfg.customInstructions,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.node.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.resolution = resolutions[i];
    });

    for (const it of items) {
      for (const cand of it.resolution.candidates) {
        batch.nodeRegistry.set(cand.id, cand);
        batch.existingNodeIds.add(cand.id);
      }
    }

    // Pass 2 - within-batch dedup over all new nodes seeded with matched-existing.
    const pass1Pairs: [Uuid, Uuid][] = items.flatMap((it) =>
      it.resolution.duplicatePairs.map((p): [Uuid, Uuid] => [
        p.extractedId,
        p.canonicalId,
      ]),
    );
    const allNewNodes = items.flatMap((it) => it.resolution.resolvedNodes);
    const matchedExistingIds = new Set(
      items.flatMap((it) => it.resolution.duplicatePairs.map((p) => p.canonicalId)),
    );
    const matchedExistingNodes = [...matchedExistingIds]
      .map((id) => batch.nodeRegistry.get(id))
      .filter((n): n is EntityNode => n !== undefined);

    const pass2Pairs = this.nodeResolutionService.dedupeAcrossBatch(
      allNewNodes,
      matchedExistingNodes,
    );

    batch.canonicalIdByNodeId = buildDirectedIdMap([...pass1Pairs, ...pass2Pairs]);
    for (const n of allNewNodes) batch.nodeRegistry.set(n.id, n);

    // Canonical nodes per episode (own-canonical + matched-existing, deduped).
    items.forEach((it) => {
      const ownCanonical = it.resolution.resolvedNodes.filter(
        (n) => (batch.canonicalIdByNodeId.get(n.id) ?? n.id) === n.id,
      );
      const matchedExisting = it.resolution.duplicatePairs
        .map((p) => {
          const canonical = batch.canonicalIdByNodeId.get(p.canonicalId) ?? p.canonicalId;
          return batch.nodeRegistry.get(canonical);
        })
        .filter((n): n is EntityNode => n !== undefined);

      const seen = new Set<Uuid>();
      it.canonicalNodes = [...ownCanonical, ...matchedExisting].filter((n) => {
        if (seen.has(n.id)) return false;
        seen.add(n.id);
        return true;
      });
    });

    // Authoritative deduped union; shares object refs with items' canonicalNodes
    // so in-place attribute/summary mutations below are visible to both.
    batch.canonicalNodes = [
      ...new Map(items.flatMap((it) => it.canonicalNodes).map((n) => [n.id, n])).values(),
    ];

    return {
      metrics: {
        'node.count.extracted': allExtractedNodes.length,
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.new': batch.canonicalNodes.filter(
          (n) => !batch.existingNodeIds.has(n.id),
        ).length,
      },
    };
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
    // Extract edges per episode using this episode's canonical nodes.
    const edgeExtractions = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      items.map(
        (it) => () =>
          this.edgeExtractionService.extractEdges(
            model,
            it.node,
            it.chunks,
            it.canonicalNodes,
            it.prevEpisodes,
            it.node.validAt,
            cfg.customInstructions,
            cfg.edgeTypes,
            cfg.effectiveEdgeTypeMappings,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.node.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.rawEdges = edgeExtractions[i].edges;
      it.chunkIndicesByEdgeId = edgeExtractions[i].chunkIndicesByEdgeId;
    });

    // Edge chunk provenance keyed by edge id. INVARIANT: episodeIndex == the
    // item's position; every [][] view handed to the resolution service below is
    // derived from `items` in order so the indices stay aligned.
    batch.chunkSources = new Map();
    items.forEach((it, i) => {
      for (const [id, indices] of it.chunkIndicesByEdgeId) {
        batch.chunkSources.set(id, { episodeIndex: i, indices });
      }
    });

    // Remap endpoints through node dedup, then embed all edges in one call.
    items.forEach((it) => {
      it.rawEdges = resolveEdgePointers(it.rawEdges, batch.canonicalIdByNodeId);
    });
    const allExtractedEdges = items.flatMap((it) => it.rawEdges);
    const allEmbeddedEdges = await this.embeddingService.embedEdges(allExtractedEdges);
    const embeddedPerItem = reassembleByOffsets(
      allEmbeddedEdges,
      items.map((it) => it.rawEdges.length),
    );
    items.forEach((it, i) => {
      it.rawEdges = embeddedPerItem[i];
    });

    // Cross-batch edge dedup -> flat canonical set. Mirrors `dedupe_edges_bulk`.
    const canonicalEdges = await this.edgeResolutionService.dedupeAcrossBatch(
      model,
      items.map((it) => it.rawEdges),
      items.map((it) => it.node),
      items.map((it) => it.chunks),
      batch.chunkSources,
      items.map((it) => it.prevEpisodes),
      cfg.customInstructions,
      ctx,
    );

    // Route each canonical edge to its ORIGIN episode so it is resolved exactly
    // once, against the episode whose validAt / previousEpisodes / chunks made it.
    for (const edge of canonicalEdges) {
      const source = batch.chunkSources.get(edge.id);
      if (!source) {
        throw new Error(`resolveEdges partition: edge ${edge.id} has no chunk source`);
      }
      items[source.episodeIndex].edgesFromThisEpisode.push(edge);
    }

    // Resolve edges per origin episode (candidates collected inside resolveEdges).
    const edgeResolutions = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      items.map(
        (it) => () =>
          this.edgeResolutionService.resolveEdges(
            model,
            items.map((other) => other.node),
            items.map((other) => other.chunks),
            batch.chunkSources,
            it.edgesFromThisEpisode,
            batch.canonicalIdByNodeId,
            it.node.validAt,
            it.prevEpisodes,
            cfg.customInstructions,
            { ...ctx, metadata: { ...ctx.metadata, episodeId: it.node.id } },
          ),
      ),
    );
    items.forEach((it, i) => {
      it.edgeResolution = edgeResolutions[i];
    });

    // Fill attributes + per-edge timestamp fallback over the freshly extracted
    // (new) edges only, so re-matched existing edges aren't re-LLM'd or
    // overwritten with thinner values from a new episode.
    const allNewEdges = items.flatMap((it) => it.edgeResolution.newEdges);
    const edgeContext = new Map<Uuid, { referenceTime: Date }>();
    items.forEach((it) => {
      for (const edge of it.edgeResolution.resolvedEdges) {
        edgeContext.set(edge.id, { referenceTime: it.node.validAt });
      }
    });

    await this.edgeExtractionService.fillEdgeAttributes(
      model,
      allNewEdges,
      batch.canonicalNodes,
      cfg.edgeTypes,
      cfg.effectiveEdgeTypeMappings,
      edgeContext,
      ctx,
    );
    await this.edgeExtractionService.extractEdgeTimestampsFallback(
      model,
      allNewEdges,
      edgeContext,
      ctx,
    );

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
      },
    };
  }

  /**
   * Phase 4 - enrich. Fills entity attributes and summaries on the canonical
   * nodes, re-embeds nodes renamed during dedup, and constructs (in memory only)
   * the saga nodes and HAS_EPISODE edges. All DB writes happen in persistPhase.
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

    const nodeContext = buildNodeContext(
      items.map((it) => it.canonicalNodes),
      items.map((it) => it.chunkIndicesByNodeId),
      batch.canonicalIdByNodeId,
      items.map((it) => it.node),
      items.map((it) => it.prevEpisodes),
      items.map((it) => it.chunks),
    );

    // Entity attributes refined from this episode's content, with resolved-edge
    // context. Runs over the full resolved set (new + matched existing).
    await this.nodeExtractionService.fillEntityAttributes(
      model,
      batch.canonicalNodes,
      allResolvedEdges,
      cfg.entityTypes,
      nodeContext,
      ctx,
    );

    // Summaries for all canonical nodes; only NEW edges as fact context so
    // matched-existing edges aren't re-emitted as known facts.
    await this.nodeExtractionService.summarizeNodes(
      model,
      batch.canonicalNodes,
      allNewEdges,
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

    // Saga construction (in-memory only). One SagaNode per distinct sagaId; one
    // HAS_EPISODE edge per episode that declares a saga. Persisted in persistPhase.
    const sagaGroups = new Map<Uuid, EpisodeWorkItem[]>();
    for (const it of items) {
      if (!it.sagaId) continue;
      sagaGroups.set(it.sagaId, [...(sagaGroups.get(it.sagaId) ?? []), it]);
    }
    for (const [sagaId, group] of sagaGroups) {
      // TODO: saga name defaults to the ID string. Plan: accept an optional
      // caller-provided name, otherwise let summarizeSaga generate one.
      batch.sagaNodes.push(
        createSagaNode({
          id: sagaId,
          name: NodeNameSchema.parse(sagaId),
          graphId: group[0].node.graphId,
        }),
      );
      for (const it of group) {
        it.hasEpisodeEdge = createHasEpisodeEdge({
          sourceNodeId: sagaId,
          targetNodeId: it.node.id,
          graphId: it.node.graphId,
        });
      }
    }

    return {
      metrics: {
        'node.count.canonical': batch.canonicalNodes.length,
        'node.count.reEmbedded': renamedNodes.length,
        'edge.count.new': allNewEdges.length,
        'saga.count': sagaGroups.size,
      },
    };
  }

  /**
   * Phase 5 - persist. Builds the MENTIONS edges, then writes everything in
   * FK-correct order: entity + episodic + saga nodes first, then entity / episodic
   * edges, then HAS_EPISODE edges last (they depend on both episodic and saga
   * nodes existing).
   */
  @Span('persistPhase', { onResult: metricsOnResult })
  private async persistPhase(
    items: EpisodeWorkItem[],
    batch: BatchState,
  ): Promise<{ metrics: SpanMetrics }> {
    items.forEach((it) => {
      it.episodicEdges = it.canonicalNodes.map((node) =>
        createEpisodicEdge({
          sourceNodeId: it.node.id,
          targetNodeId: node.id,
          graphId: it.node.graphId,
        }),
      );
    });

    const allResolvedEdges = items.flatMap((it) => it.edgeResolution.resolvedEdges);
    const allInvalidatedEdges = items.flatMap((it) => it.edgeResolution.invalidatedEdges);
    const allEpisodicEdges = items.flatMap((it) => it.episodicEdges);
    const hasEpisodeEdges = items
      .map((it) => it.hasEpisodeEdge)
      .filter((e): e is HasEpisodeEdge => e !== undefined);

    // Nodes first. Postgres FK constraints reject edges whose endpoints don't
    // yet exist. Saga nodes are upserts (may pre-exist from earlier batches).
    await Promise.all([
      this.entityNodeRepository.saveBulk(batch.canonicalNodes),
      this.episodicNodeRepository.saveBulk(items.map((it) => it.node)),
      ...batch.sagaNodes.map((saga) => this.sagaNodeRepository.createIfNotExists(saga)),
    ]);

    // Entity + episodic edges, then HAS_EPISODE edges (FK onto episodic + saga).
    await Promise.all([
      this.entityEdgeRepository.saveBulk(allResolvedEdges),
      this.entityEdgeRepository.saveBulk(allInvalidatedEdges),
      this.episodicEdgeRepository.saveBulk(allEpisodicEdges),
    ]);
    await Promise.all(
      hasEpisodeEdges.map((edge) => this.hasEpisodeEdgeRepository.save(edge)),
    );

    return {
      metrics: {
        'node.count.persisted': batch.canonicalNodes.length,
        'episode.count': items.length,
        'saga.count': batch.sagaNodes.length,
        'edge.count.resolved': allResolvedEdges.length,
        'edge.count.invalidated': allInvalidatedEdges.length,
        'hasEpisodeEdge.count': hasEpisodeEdges.length,
      },
    };
  }
}
