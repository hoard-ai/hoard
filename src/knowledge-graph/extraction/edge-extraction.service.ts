import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';
import { z } from 'zod';

import { Uuid } from '@/common/schemas';
import { invokeStructured } from '@/llm';
import {
  LLM_TRACER,
  type LlmContext,
  type LlmTracer,
  metricsOnResult,
  Span,
  type SpanMetrics,
} from '@/observability';

import { getApplicableEdgeTypes } from '../episode/episode-utils';
import type { EdgeTypeMap, EdgeTypeMappings } from '../episode/types';
import { createEntityEdge, EntityEdge, EntityNode, type EpisodicNode } from '../models';
import {
  buildEnrichEdgeMessages,
  buildEnrichEdgeSchema,
  buildEnrichEdgeValidator,
  buildExtractEdgesMessages,
  buildExtractEdgesValidator,
  type ExtractedEdgesOutput,
  ExtractedEdgesSchema,
  ExtractedEdgesWithCorefSchema,
} from '../prompts';
import {
  type ScopedCandidate,
  selectCandidatesUpToChunk,
  selectCoreferencesForChunks,
} from '../prompts/coref-utils';
import { selectChunkText } from '../prompts/text-utils';
import type { EdgeChunkSources } from '../resolution/types';
import {
  type CommittedCorefBinding,
  ExtractEdgesResult,
  type TrackedUnresolvedReference,
} from './types';

@Injectable()
export class EdgeExtractionService {
  constructor(@Inject(LLM_TRACER) private readonly llmTracer: LlmTracer) {}

  async extractEdges(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    nodes: EntityNode[],
    previousEpisodes: EpisodicNode[],
    customInstructions?: string,
    edgeTypes?: EdgeTypeMap,
    edgeTypeMappings?: EdgeTypeMappings,
    resolveCoreferences = false,
    corefCandidates: ScopedCandidate[] = [],
    unresolvedReferences: TrackedUnresolvedReference[] = [],
    ctx?: LlmContext,
  ): Promise<ExtractEdgesResult> {
    const { metrics: _m, ...rest } = await this.extractEdgesImpl(
      model,
      episode,
      chunks,
      nodes,
      previousEpisodes,
      customInstructions,
      edgeTypes,
      edgeTypeMappings,
      resolveCoreferences,
      corefCandidates,
      unresolvedReferences,
      ctx,
    );
    return rest;
  }

  @Span('edgeExtraction', { onResult: metricsOnResult })
  private async extractEdgesImpl(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    nodes: EntityNode[],
    previousEpisodes: EpisodicNode[],
    customInstructions: string | undefined,
    edgeTypes: EdgeTypeMap | undefined,
    edgeTypeMappings: EdgeTypeMappings | undefined,
    resolveCoreferences: boolean,
    corefCandidates: ScopedCandidate[],
    unresolvedReferences: TrackedUnresolvedReference[],
    ctx: LlmContext | undefined,
  ): Promise<ExtractEdgesResult & { metrics: SpanMetrics }> {
    // Each chunk gets the SAME full canonical node list, so the entity index
    // space is shared across chunks (nodes[idx] resolves identically everywhere).
    const perChunk = await Promise.all(
      chunks.map(async (chunk, chunkIdx) => {
        // Candidate antecedents introduced at or before this chunk (prefix rule).
        const coreferences = resolveCoreferences
          ? selectCandidatesUpToChunk(corefCandidates, chunkIdx)
          : [];
        // In-view rule: only this chunk's own unresolved references are claimable
        // here. This array's order is the refIdx space the model echoes back.
        const chunkUnresolved = resolveCoreferences
          ? unresolvedReferences.filter((r) => r.sourceChunkIndex === chunkIdx)
          : [];
        const messages = buildExtractEdgesMessages({
          episode: { ...episode, content: chunk },
          nodes,
          previousEpisodes,
          customInstructions,
          edgeTypes,
          edgeTypeMappings,
          coreferences,
          unresolvedReferences: chunkUnresolved,
        });
        const opts = {
          callbacks: this.llmTracer.getCallbacks(ctx),
          runName: 'extract-edges',
          tags: ['knowledge-graph', 'extraction.edge'],
          validate: buildExtractEdgesValidator({
            nodes,
            unresolvedReferences: chunkUnresolved,
          }),
        };

        let rawEdges: ExtractedEdgesOutput['edges'];
        const bindings: CommittedCorefBinding[] = [];
        if (resolveCoreferences) {
          const result = await invokeStructured(
            model,
            ExtractedEdgesWithCorefSchema,
            messages,
            opts,
          );
          rawEdges = result.edges;
          for (const used of result.usedCoreferences) {
            // Both indices are validated in range (buildExtractEdgesValidator), so
            // the lookups are total - index directly, as the edge endpoints do.
            // A null idx is a genuine non-claim; a non-null idx MUST resolve.
            let resolvedUnresolvedReferenceId: Uuid | null = null;
            if (used.unresolvedReferenceIdx !== null) {
              const claimedRef = chunkUnresolved[used.unresolvedReferenceIdx];
              if (!claimedRef) {
                throw new Error(
                  `extractEdges: unresolvedReferenceIdx ${used.unresolvedReferenceIdx} out of range for chunk ${chunkIdx} (${chunkUnresolved.length} refs)`,
                );
              }
              resolvedUnresolvedReferenceId = claimedRef.id;
            }
            bindings.push({
              surfaceForm: used.surfaceForm,
              boundNodeId: nodes[used.entityIdx].id,
              sourceChunkIndex: chunkIdx,
              locatingQuote: used.locatingQuote,
              resolvedUnresolvedReferenceId,
            });
          }
        } else {
          const result = await invokeStructured(
            model,
            ExtractedEdgesSchema,
            messages,
            opts,
          );
          rawEdges = result.edges;
        }

        // Timestamps are NOT extracted here - they are filled later, per edge and
        // chunk-grounded, by enrichEdges. Edges start with null validAt/invalidAt.
        const chunkEdges = rawEdges.map((e) =>
          createEntityEdge({
            name: e.relationType,
            fact: e.fact,
            graphId: episode.graphId,
            sourceNodeId: nodes[e.sourceEntityIdx].id,
            targetNodeId: nodes[e.targetEntityIdx].id,
            episodes: [episode.id],
          }),
        );
        return { chunkEdges, bindings };
      }),
    );
    // Flatten chunk edges into one per-episode list; tag each with its
    // originating chunk index (singleton until dedup unions duplicates).
    const edges: EntityEdge[] = [];
    const chunkIndicesByEdgeId = new Map<Uuid, Set<number>>();
    const committedCorefBindings: CommittedCorefBinding[] = [];

    perChunk.forEach(({ chunkEdges, bindings }, chunkIdx) => {
      for (const edge of chunkEdges) {
        edges.push(edge);
        chunkIndicesByEdgeId.set(edge.id, new Set([chunkIdx]));
      }
      committedCorefBindings.push(...bindings);
    });
    return {
      edges,
      chunkIndicesByEdgeId,
      committedCorefBindings,
      metrics: {
        'episode.id': episode.id,
        'chunks.count': chunks.length,
        'nodes.input.count': nodes.length,
        'edgeTypes.count': edgeTypes ? Object.keys(edgeTypes).length : 0,
        'edges.extracted.count': edges.length,
        'coref.bindings.count': committedCorefBindings.length,
      },
    };
  }

  /**
   * Unified edge enrichment. Runs one LLM call per new edge to fill its
   * temporal bounds (validAt/invalidAt) and, when the edge has a custom fact
   * type, its typed attributes - both grounded in the edge's own chunk text.
   * Mutates the edges in place. Runs on EVERY new edge (typed and
   * untyped), so it must precede invalidation, which depends on the bounds.
   */
  async enrichEdges(
    model: BaseChatModel,
    newEdges: EntityEdge[],
    canonicalNodes: EntityNode[],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    edgeTypes?: EdgeTypeMap,
    edgeTypeMappings?: EdgeTypeMappings,
    committedCorefBindingsPerEpisode: CommittedCorefBinding[][] = [],
    ctx?: LlmContext,
  ): Promise<void> {
    await this.enrichEdgesImpl(
      model,
      newEdges,
      canonicalNodes,
      episodes,
      chunksPerEpisode,
      chunkSources,
      edgeTypes,
      edgeTypeMappings,
      committedCorefBindingsPerEpisode,
      ctx,
    );
  }

  @Span('enrichEdges', { onResult: metricsOnResult })
  private async enrichEdgesImpl(
    model: BaseChatModel,
    newEdges: EntityEdge[],
    canonicalNodes: EntityNode[],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    edgeTypes: EdgeTypeMap | undefined,
    edgeTypeMappings: EdgeTypeMappings | undefined,
    committedCorefBindingsPerEpisode: CommittedCorefBinding[][],
    ctx: LlmContext | undefined,
  ): Promise<{ metrics: SpanMetrics }> {
    const idToNode = new Map<Uuid, EntityNode>(canonicalNodes.map((n) => [n.id, n]));
    const nodeNameById = new Map(canonicalNodes.map((n) => [n.id, n.name]));
    let typedCount = 0;

    await Promise.all(
      newEdges.map(async (edge) => {
        const source = chunkSources.get(edge.id);
        if (!source) {
          throw new Error(
            `enrichEdges: edge ${edge.id} has no originating chunk indices`,
          );
        }
        const episode: EpisodicNode = {
          ...episodes[source.episodeIndex],
          content: selectChunkText(source.indices, chunksPerEpisode[source.episodeIndex]),
        };
        const referenceTime = episodes[source.episodeIndex].validAt;
        const coreferences = selectCoreferencesForChunks(
          committedCorefBindingsPerEpisode[source.episodeIndex] ?? [],
          nodeNameById,
          source.indices,
        );

        // Custom fact-type schema, when the edge's relation maps to one for its
        // endpoint labels. Untyped edges get temporal-only enrichment.
        let customSchema: z.ZodType | undefined = undefined;
        if (edgeTypes && edgeTypeMappings) {
          const src = idToNode.get(edge.sourceNodeId);
          const tgt = idToNode.get(edge.targetNodeId);
          if (!src || !tgt) {
            throw new Error(
              `enrichEdges: edge ${edge.id} endpoint missing from canonical nodes`,
            );
          }
          const applicable = getApplicableEdgeTypes(
            src.labels,
            tgt.labels,
            edgeTypes,
            edgeTypeMappings,
          );
          customSchema = applicable[edge.name]?.schema;
        }
        const hasCustomAttributes = customSchema !== undefined;
        if (hasCustomAttributes) typedCount++;

        const result = await invokeStructured(
          model,
          buildEnrichEdgeSchema(customSchema),
          buildEnrichEdgeMessages({
            fact: edge.fact,
            episode,
            referenceTime,
            existingAttributes: edge.attributes,
            hasCustomAttributes,
            coreferences,
            labelChunks: chunksPerEpisode[source.episodeIndex].length > 1,
          }),
          {
            callbacks: this.llmTracer.getCallbacks(ctx),
            runName: 'enrich-edge',
            tags: ['knowledge-graph', 'enrich.edge'],
            validate: buildEnrichEdgeValidator(),
          },
        );

        // ISO string validated at schema level
        if (result.validAt) edge.validAt = new Date(result.validAt);
        if (result.invalidAt) edge.invalidAt = new Date(result.invalidAt);
        if (result.attributes) {
          edge.attributes = { ...edge.attributes, ...result.attributes };
        }
      }),
    );

    return {
      metrics: {
        'new.count': newEdges.length,
        'typed.count': typedCount,
      },
    };
  }
}
