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
import { EdgeTypeMap, EdgeTypeMappings } from '../episode/types';
import { createEntityEdge, EntityEdge, EntityNode, EpisodicNode } from '../models';
import {
  buildExtractEdgesMessages,
  buildExtractEdgesValidator,
  buildExtractTimestampsMessages,
  buildExtractTimestampsValidator,
  buildFillEdgeAttributesMessages,
  EdgeTimestampsSchema,
  ExtractedEdgesSchema,
} from '../prompts';
import { EdgeReferenceTimeContext, ExtractEdgesResult } from './types';

@Injectable()
export class EdgeExtractionService {
  constructor(@Inject(LLM_TRACER) private readonly llmTracer: LlmTracer) {}

  async extractEdges(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    nodes: EntityNode[],
    previousEpisodes: EpisodicNode[],
    referenceTime: Date,
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
      referenceTime,
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
    referenceTime: Date,
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
          referenceTime,
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
        return result.edges.map((e) =>
          createEntityEdge({
            name: e.relationType,
            fact: e.fact,
            graphId: episode.graphId,
            sourceNodeId: nodes[e.sourceEntityIdx].id,
            targetNodeId: nodes[e.targetEntityIdx].id,
            episodes: [episode.id],
            validAt: e.validAt ? new Date(e.validAt) : null,
            invalidAt: e.invalidAt ? new Date(e.invalidAt) : null,
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

  async fillEdgeAttributes(
    model: BaseChatModel,
    resolvedEdges: EntityEdge[],
    canonicalNodes: EntityNode[],
    edgeTypes: EdgeTypeMap | undefined,
    edgeTypeMappings: EdgeTypeMappings | undefined,
    edgeContext: EdgeReferenceTimeContext,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.fillEdgeAttributesImpl(
      model,
      resolvedEdges,
      canonicalNodes,
      edgeTypes,
      edgeTypeMappings,
      edgeContext,
      ctx,
    );
  }

  @Span('fillEdgeAttributes', { onResult: metricsOnResult })
  private async fillEdgeAttributesImpl(
    model: BaseChatModel,
    resolvedEdges: EntityEdge[],
    canonicalNodes: EntityNode[],
    edgeTypes: EdgeTypeMap | undefined,
    edgeTypeMappings: EdgeTypeMappings | undefined,
    edgeContext: EdgeReferenceTimeContext,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const baseMetrics: SpanMetrics = {
      'edges.count': resolvedEdges.length,
      'edgeTypes.count': edgeTypes ? Object.keys(edgeTypes).length : 0,
    };

    if (!edgeTypes || !edgeTypeMappings) {
      return { metrics: { ...baseMetrics, 'extracted.count': 0 } };
    }
    const idToNode = new Map<Uuid, EntityNode>(canonicalNodes.map((n) => [n.id, n]));

    type EdgeAttrTask = {
      edge: EntityEdge;
      schema: z.ZodType;
      referenceTime: Date;
    };
    const tasks: EdgeAttrTask[] = [];

    for (const edge of resolvedEdges) {
      const src = idToNode.get(edge.sourceNodeId);
      const tgt = idToNode.get(edge.targetNodeId);
      if (!src || !tgt) {
        throw new Error(
          `fillEdgeAttributes: edge ${edge.id} endpoint missing from canonical nodes`,
        );
      }
      const applicable = getApplicableEdgeTypes(
        src.labels,
        tgt.labels,
        edgeTypes,
        edgeTypeMappings,
      );
      // Edges whose relation isn't a custom fact type get no attributes.
      const typeDef = applicable[edge.name];
      if (!typeDef) continue;

      const edgeCtx = edgeContext.get(edge.id);
      if (!edgeCtx) {
        throw new Error(`fillEdgeAttributes: edge ${edge.id} missing from edgeContext`);
      }
      tasks.push({ edge, schema: typeDef.schema, referenceTime: edgeCtx.referenceTime });
    }

    await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      tasks.map(({ edge, schema, referenceTime }) => async () => {
        const attrs = (await invokeStructured(
          model,
          schema,
          buildFillEdgeAttributesMessages({
            fact: edge.fact,
            referenceTime,
            existingAttributes: edge.attributes ?? {},
          }),
          {
            callbacks: this.llmTracer.getCallbacks(ctx),
            runName: 'fill-edge-attributes',
            tags: ['knowledge-graph', 'attributes.edge'],
          },
        )) as Record<string, unknown>;
        edge.attributes = { ...edge.attributes, ...attrs };
      }),
    );
    return { metrics: { ...baseMetrics, 'extracted.count': tasks.length } };
  }

  // Per-edge fallback: when the batch extraction prompt leaves an edge with
  // both validAt and invalidAt null, ask the LLM specifically for the temporal
  // window of that single fact. Mirrors graphiti's `_extract_edge_timestamps`
  // (edge_operations.py:576).
  async extractEdgeTimestampsFallback(
    model: BaseChatModel,
    edges: EntityEdge[],
    edgeContext: EdgeReferenceTimeContext,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.extractEdgeTimestampsFallbackImpl(model, edges, edgeContext, ctx);
  }

  @Span('extractEdgeTimestampsFallback', { onResult: metricsOnResult })
  private async extractEdgeTimestampsFallbackImpl(
    model: BaseChatModel,
    edges: EntityEdge[],
    edgeContext: EdgeReferenceTimeContext,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const candidates = edges.filter(
      (e) => e.validAt === null && e.invalidAt === null && edgeContext.has(e.id),
    );

    await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      candidates.map((edge) => async () => {
        const referenceTime = edgeContext.get(edge.id)!.referenceTime;
        const result = await invokeStructured(
          model,
          EdgeTimestampsSchema,
          buildExtractTimestampsMessages({ fact: edge.fact, referenceTime }),
          {
            callbacks: this.llmTracer.getCallbacks(ctx),
            runName: 'extract-edge-timestamps-fallback',
            tags: ['knowledge-graph', 'timestamps.edge.fallback'],
            validate: buildExtractTimestampsValidator(),
          },
        );

        if (result.validAt) edge.validAt = new Date(result.validAt);
        if (result.invalidAt) edge.invalidAt = new Date(result.invalidAt);
      }),
    );

    return {
      metrics: { 'edges.count': edges.length, 'candidates.count': candidates.length },
    };
  }
}
