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

import { LLM_CONCURRENCY_LIMIT, withConcurrency } from '../batch-utils';
import { getApplicableEdgeTypes } from '../episode/episode-utils';
import type { EdgeTypeMap, EdgeTypeMappings } from '../episode/types';
import { createEntityEdge, EntityEdge, EntityNode, type EpisodicNode } from '../models';
import {
  buildEnrichEdgeMessages,
  buildEnrichEdgeSchema,
  buildEnrichEdgeValidator,
  buildExtractEdgesMessages,
  buildExtractEdgesValidator,
  ExtractedEdgesSchema,
} from '../prompts';
import { selectChunkText } from '../prompts/text-utils';
import type { EdgeChunkSources } from '../resolution/types';
import { ExtractEdgesResult } from './types';

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
    ctx?: LlmContext,
  ): Promise<ExtractEdgesResult> {
    const { edges, chunkIndicesByEdgeId } = await this.extractEdgesImpl(
      model,
      episode,
      chunks,
      nodes,
      previousEpisodes,
      customInstructions,
      edgeTypes,
      edgeTypeMappings,
      ctx,
    );
    return { edges, chunkIndicesByEdgeId };
  }

  @Span('edgeExtraction', { onResult: metricsOnResult })
  private async extractEdgesImpl(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    nodes: EntityNode[],
    previousEpisodes: EpisodicNode[],
    customInstructions?: string,
    edgeTypes?: EdgeTypeMap,
    edgeTypeMappings?: EdgeTypeMappings,
    ctx?: LlmContext,
  ): Promise<{
    edges: EntityEdge[];
    chunkIndicesByEdgeId: Map<Uuid, Set<number>>;
    metrics: SpanMetrics;
  }> {
    // Each chunk gets the SAME full canonical node list, so the entity index
    // space is shared across chunks (nodes[idx] resolves identically everywhere).
    const perChunk = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      chunks.map((chunk) => async () => {
        const messages = buildExtractEdgesMessages({
          episode: { ...episode, content: chunk },
          nodes,
          previousEpisodes,
          customInstructions,
          edgeTypes,
          edgeTypeMappings,
        });
        const result = await invokeStructured(model, ExtractedEdgesSchema, messages, {
          callbacks: this.llmTracer.getCallbacks(ctx),
          runName: 'extract-edges',
          tags: ['knowledge-graph', 'extraction.edge'],
          validate: buildExtractEdgesValidator({ nodes }),
        });
        // Timestamps are NOT extracted here - they are filled later, per edge and
        // chunk-grounded, by enrichEdges. Edges start with null validAt/invalidAt.
        return result.edges.map((e) =>
          createEntityEdge({
            name: e.relationType,
            fact: e.fact,
            graphId: episode.graphId,
            sourceNodeId: nodes[e.sourceEntityIdx].id,
            targetNodeId: nodes[e.targetEntityIdx].id,
            episodes: [episode.id],
          }),
        );
      }),
    );
    // Flatten chunk edges into one per-episode list; tag each with its
    // originating chunk index (singleton until dedup unions duplicates).
    const edges: EntityEdge[] = [];
    const chunkIndicesByEdgeId = new Map<Uuid, Set<number>>();

    perChunk.forEach((chunkEdges, chunkIdx) => {
      for (const edge of chunkEdges) {
        edges.push(edge);
        chunkIndicesByEdgeId.set(edge.id, new Set([chunkIdx]));
      }
    });
    return {
      edges,
      chunkIndicesByEdgeId,
      metrics: {
        'episode.id': episode.id,
        'chunks.count': chunks.length,
        'nodes.input.count': nodes.length,
        'edgeTypes.count': edgeTypes ? Object.keys(edgeTypes).length : 0,
        'edges.extracted.count': edges.length,
      },
    };
  }

  /**
   * Unified edge enrichment. Runs one LLM call per surviving edge to fill its
   * temporal bounds (validAt/invalidAt) and, when the edge has a custom fact
   * type, its typed attributes - both grounded in the edge's own chunk text.
   * Mutates the survivor edges in place. Runs on EVERY survivor (typed and
   * untyped), so it must precede invalidation, which depends on the bounds.
   */
  async enrichEdges(
    model: BaseChatModel,
    survivors: EntityEdge[],
    canonicalNodes: EntityNode[],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    edgeTypes?: EdgeTypeMap,
    edgeTypeMappings?: EdgeTypeMappings,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.enrichEdgesImpl(
      model,
      survivors,
      canonicalNodes,
      episodes,
      chunksPerEpisode,
      chunkSources,
      edgeTypes,
      edgeTypeMappings,
      ctx,
    );
  }

  @Span('enrichEdges', { onResult: metricsOnResult })
  private async enrichEdgesImpl(
    model: BaseChatModel,
    survivors: EntityEdge[],
    canonicalNodes: EntityNode[],
    episodes: EpisodicNode[],
    chunksPerEpisode: string[][],
    chunkSources: EdgeChunkSources,
    edgeTypes?: EdgeTypeMap,
    edgeTypeMappings?: EdgeTypeMappings,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const idToNode = new Map<Uuid, EntityNode>(canonicalNodes.map((n) => [n.id, n]));
    let typedCount = 0;

    await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      survivors.map((edge) => async () => {
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
        'survivors.count': survivors.length,
        'typed.count': typedCount,
      },
    };
  }
}
