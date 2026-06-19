import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

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
import type { EntityTypeMap } from '../episode/types';
import { createEntityNode, EntityEdge, EntityNode, type EpisodicNode } from '../models';
import {
  buildExtractNodesMessages,
  buildExtractNodesValidator,
  buildFillEntityAttributesMessages,
  buildNodeSummaryMessages,
  buildNodeSummaryValidator,
  ExtractedEntitiesSchema,
  NodeSummarySchema,
} from '../prompts';
import { selectChunkText } from '../prompts/text-utils';
import { NodeLabel, NodeLabels, NodeLabelSchema } from '../types';
import {
  ExtractNodesResult,
  MAX_NODES_PER_SUMMARY_BATCH,
  type NodeEpisodeContext,
} from './types';

function resolveLabels(
  entityTypeId: number | undefined,
  entityTypes?: EntityTypeMap,
): NodeLabels {
  const entity = NodeLabelSchema.parse('Entity');

  if (entityTypeId === undefined || !entityTypes) {
    return [entity];
  }
  const labels = Object.keys(entityTypes) as NodeLabel[];
  return [entity, labels[entityTypeId]];
}

@Injectable()
export class NodeExtractionService {
  constructor(@Inject(LLM_TRACER) private readonly llmTracer: LlmTracer) {}

  async extractNodes(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    previousEpisodes: EpisodicNode[],
    entityTypes?: EntityTypeMap,
    customInstructions?: string,
    excludedEntityTypes?: string[],
    ctx?: LlmContext,
  ): Promise<ExtractNodesResult> {
    const { extractedNodes, chunkIndicesByNodeId } = await this.extractNodesImpl(
      model,
      episode,
      chunks,
      previousEpisodes,
      entityTypes,
      customInstructions,
      excludedEntityTypes,
      ctx,
    );
    return { nodes: extractedNodes, chunkIndicesByNodeId };
  }

  @Span('extractNodes', { onResult: metricsOnResult })
  private async extractNodesImpl(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    previousEpisodes: EpisodicNode[],
    entityTypes?: EntityTypeMap,
    customInstructions?: string,
    excludedEntityTypes?: string[],
    ctx?: LlmContext,
  ): Promise<{
    extractedNodes: EntityNode[];
    chunkIndicesByNodeId: Map<Uuid, Set<number>>;
    metrics: SpanMetrics;
  }> {
    const perChunk = await withConcurrency(
      LLM_CONCURRENCY_LIMIT,
      chunks.map(
        (chunk) => () =>
          this.extractNodesFromChunk(
            model,
            { ...episode, content: chunk },
            previousEpisodes,
            entityTypes,
            customInstructions,
            excludedEntityTypes,
            ctx,
          ),
      ),
    );
    // Deduplicate nodes across chunks by case-insensitive name (first occurrence
    // wins). The kept node unions the chunk indices its name appeared in, so it
    // owns every chunk it was extracted from. No-op when there's a single chunk.
    const nodesByName = new Map<string, EntityNode>();
    const chunkIndicesByNodeId = new Map<Uuid, Set<number>>();

    perChunk.forEach((nodes, chunkIdx) => {
      for (const node of nodes) {
        const key = node.name.toLowerCase();
        const existing = nodesByName.get(key);

        if (!existing) {
          nodesByName.set(key, node);
          chunkIndicesByNodeId.set(node.id, new Set([chunkIdx]));
        } else {
          chunkIndicesByNodeId.get(existing.id)!.add(chunkIdx);
        }
      }
    });
    const extractedNodes = [...nodesByName.values()];

    return {
      extractedNodes,
      chunkIndicesByNodeId,
      metrics: {
        'episode.id': episode.id,
        'entityTypes.count': entityTypes ? Object.keys(entityTypes).length : 0,
        'chunks.count': chunks.length,
        'extracted.count': extractedNodes.length,
      },
    };
  }

  private async extractNodesFromChunk(
    model: BaseChatModel,
    episode: EpisodicNode,
    previousEpisodes: EpisodicNode[],
    entityTypes: EntityTypeMap | undefined,
    customInstructions: string | undefined,
    excludedEntityTypes: string[] | undefined,
    ctx: LlmContext | undefined,
  ): Promise<EntityNode[]> {
    const messages = buildExtractNodesMessages({
      episode,
      previousEpisodes,
      entityTypes,
      customInstructions,
    });
    const result = await invokeStructured(model, ExtractedEntitiesSchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'extract-nodes',
      tags: ['knowledge-graph', 'extraction.node'],
      validate: buildExtractNodesValidator({ entityTypes }),
    });

    return result.extractedEntities
      .filter((e) => e.name.trim() !== '')
      .map((e) =>
        createEntityNode({
          name: e.name,
          graphId: episode.graphId,
          labels: resolveLabels(e.entityTypeId, entityTypes),
        }),
      )
      .filter((node) => {
        if (!excludedEntityTypes?.length) return true;
        const specificLabel = node.labels.find((l) => l !== 'Entity') ?? 'Entity';
        return !excludedEntityTypes.includes(specificLabel);
      });
  }

  async fillEntityAttributes(
    model: BaseChatModel,
    nodes: EntityNode[],
    allEdges: EntityEdge[],
    entityTypes: EntityTypeMap | undefined,
    nodeContext: NodeEpisodeContext,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.fillEntityAttributesImpl(
      model,
      nodes,
      allEdges,
      entityTypes,
      nodeContext,
      ctx,
    );
  }

  @Span('fillEntityAttributes', { onResult: metricsOnResult })
  private async fillEntityAttributesImpl(
    model: BaseChatModel,
    nodes: EntityNode[],
    allEdges: EntityEdge[],
    entityTypes: EntityTypeMap | undefined,
    nodeContext: NodeEpisodeContext,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    const baseMetrics: SpanMetrics = {
      'nodes.count': nodes.length,
      'entityTypes.count': entityTypes ? Object.keys(entityTypes).length : 0,
    };
    if (!entityTypes) return { metrics: { ...baseMetrics, 'extracted.count': 0 } };
    const tasks: Array<() => Promise<void>> = [];

    for (const node of nodes) {
      const label = node.labels.find((l) => l !== 'Entity');
      const entityType = label ? entityTypes[label] : undefined;
      if (!entityType) continue;

      const nodeCtx = nodeContext.get(node.id);
      if (!nodeCtx) {
        throw new Error(`fillEntityAttributes: node ${node.id} missing from nodeContext`);
      }
      const nodeEdges = allEdges.filter(
        (e) => e.sourceNodeId === node.id || e.targetNodeId === node.id,
      );

      tasks.push(async () => {
        const attrMessages = buildFillEntityAttributesMessages({
          entityName: node.name,
          episodeContent: selectChunkText(nodeCtx.sourceChunkIndices, nodeCtx.chunks),
          previousEpisodesContent: nodeCtx.previousEpisodes.map((ep) => ep.content),
          relatedFacts: nodeEdges.map((e) => e.fact),
          referenceTime: nodeCtx.episode.validAt,
          existingAttributes: node.attributes,
        });
        const attrs = (await invokeStructured(model, entityType.schema, attrMessages, {
          callbacks: this.llmTracer.getCallbacks(ctx),
          runName: 'fill-entity-attributes',
          tags: ['knowledge-graph', 'attributes.entity'],
        })) as Record<string, unknown>;

        node.attributes = { ...node.attributes, ...attrs };
      });
    }
    await withConcurrency(LLM_CONCURRENCY_LIMIT, tasks);

    return { metrics: { ...baseMetrics, 'extracted.count': tasks.length } };
  }

  async summarizeNodes(
    model: BaseChatModel,
    nodes: EntityNode[],
    allEdges: EntityEdge[],
    entityTypes: EntityTypeMap | undefined,
    nodeContext: NodeEpisodeContext,
    ctx?: LlmContext,
  ): Promise<void> {
    await this.summarizeNodesImpl(model, nodes, allEdges, entityTypes, nodeContext, ctx);
  }

  @Span('summarizeNodes', { onResult: metricsOnResult })
  private async summarizeNodesImpl(
    model: BaseChatModel,
    nodes: EntityNode[],
    allEdges: EntityEdge[],
    entityTypes: EntityTypeMap | undefined,
    nodeContext: NodeEpisodeContext,
    ctx?: LlmContext,
  ): Promise<{ metrics: SpanMetrics }> {
    if (nodes.length === 0) {
      return { metrics: { 'nodes.count': 0, 'summarized.count': 0 } };
    }

    // Group nodes by their originating episode so each node is summarized with its own context.
    const nodesByEpisode = new Map<
      Uuid,
      {
        episode: EpisodicNode;
        previousEpisodes: EpisodicNode[];
        chunks: string[];
        nodes: EntityNode[];
      }
    >();

    for (const node of nodes) {
      const nodeCtx = nodeContext.get(node.id);
      if (!nodeCtx) {
        throw new Error(`summarizeNodes: node ${node.id} missing from nodeContext`);
      }
      const entry = nodesByEpisode.get(nodeCtx.episode.id);

      if (entry) {
        entry.nodes.push(node);
      } else {
        nodesByEpisode.set(nodeCtx.episode.id, {
          episode: nodeCtx.episode,
          previousEpisodes: nodeCtx.previousEpisodes,
          chunks: nodeCtx.chunks,
          nodes: [node],
        });
      }
    }
    const entityTypeDescriptions: Record<string, string> = entityTypes
      ? Object.fromEntries(
          Object.entries(entityTypes).map(([label, { description }]) => [
            label,
            description,
          ]),
        )
      : {};

    const summaryMap = new Map<string, string>();
    const tasks: Array<() => Promise<void>> = [];

    for (const {
      episode,
      previousEpisodes,
      chunks,
      nodes: groupNodes,
    } of nodesByEpisode.values()) {
      const summaryInput = groupNodes.map((n) => {
        const label = n.labels.find((l) => l !== 'Entity');
        const type = label && entityTypes?.[label] ? label : undefined;
        return {
          name: n.name,
          type,
          existingSummary: n.summary,
          facts: allEdges
            .filter((e) => e.sourceNodeId === n.id || e.targetNodeId === n.id)
            .map((e) => e.fact),
        };
      });

      for (let i = 0; i < summaryInput.length; i += MAX_NODES_PER_SUMMARY_BATCH) {
        const batch = summaryInput.slice(i, i + MAX_NODES_PER_SUMMARY_BATCH);
        const batchNodes = groupNodes.slice(i, i + MAX_NODES_PER_SUMMARY_BATCH);

        tasks.push(async () => {
          // Episode text = the chunks this batch's nodes came from (union). The
          // grouping loop above throws on a missing node, so the lookup is safe.
          const batchChunkIndices = new Set<number>();
          for (const n of batchNodes) {
            for (const idx of nodeContext.get(n.id)!.sourceChunkIndices) {
              batchChunkIndices.add(idx);
            }
          }
          const episodeText = selectChunkText(batchChunkIndices, chunks);

          const summaryMessages = buildNodeSummaryMessages({
            episode: { ...episode, content: episodeText },
            previousEpisodes,
            nodes: batch,
            entityTypeDescriptions,
          });
          const summaryResult = await invokeStructured(
            model,
            NodeSummarySchema,
            summaryMessages,
            {
              callbacks: this.llmTracer.getCallbacks(ctx),
              runName: 'summarize-nodes',
              tags: ['knowledge-graph', 'node.summary'],
              validate: buildNodeSummaryValidator({ nodes: batch }),
            },
          );
          for (const s of summaryResult.summaries) {
            summaryMap.set(s.name, s.summary);
          }
        });
      }
    }
    await withConcurrency(LLM_CONCURRENCY_LIMIT, tasks);

    for (const node of nodes) {
      const summary = summaryMap.get(node.name);
      if (summary !== undefined) node.summary = summary;
    }
    return {
      metrics: {
        'nodes.count': nodes.length,
        'summarized.count': summaryMap.size,
      },
    };
  }
}
