import { randomUUID } from 'node:crypto';

import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

import { Uuid, UuidSchema } from '@/common/schemas';
import { KnowledgeGraphConfigService } from '@/config/knowledge-graph';
import { invokeStructured } from '@/llm';
import {
  LLM_TRACER,
  type LlmContext,
  type LlmTracer,
  metricsOnResult,
  Span,
  type SpanMetrics,
} from '@/observability';

import { withConcurrency } from '../batch-utils';
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
import { selectCoreferencesForChunks } from '../prompts/coref-utils';
import { selectChunkText } from '../prompts/text-utils';
import { NodeLabel, NodeLabels, NodeLabelSchema } from '../types';
import {
  type EntityCorefDescriptor,
  ExtractNodesResult,
  MAX_NODES_PER_SUMMARY_BATCH,
  type NodeEpisodeContext,
  type TrackedUnresolvedReference,
  type UnresolvedReference,
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
  constructor(
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
    private readonly kgConfig: KnowledgeGraphConfigService,
  ) {}

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
    const { metrics: _m, ...rest } = await this.extractNodesImpl(
      model,
      episode,
      chunks,
      previousEpisodes,
      entityTypes,
      customInstructions,
      excludedEntityTypes,
      ctx,
    );
    return rest;
  }

  @Span('extractNodes', { onResult: metricsOnResult })
  private async extractNodesImpl(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    previousEpisodes: EpisodicNode[],
    entityTypes: EntityTypeMap | undefined,
    customInstructions: string | undefined,
    excludedEntityTypes: string[] | undefined,
    ctx: LlmContext | undefined,
  ): Promise<ExtractNodesResult & { metrics: SpanMetrics }> {
    const perChunk = await Promise.all(
      chunks.map((chunk) =>
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
    // owns every chunk it was extracted from. Coref descriptors merge onto the
    // kept node: union aliases + observed pronouns, keep the first
    // identifyingDescription. No-op for a single chunk.
    const nodesByName = new Map<string, EntityNode>();
    const chunkIndicesByExtractedId = new Map<Uuid, Set<number>>();
    const corefByExtractedId = new Map<Uuid, EntityCorefDescriptor>();
    const unresolvedReferences: TrackedUnresolvedReference[] = [];

    perChunk.forEach((chunk, chunkIdx) => {
      for (const ref of chunk.unresolvedReferences) {
        unresolvedReferences.push({
          ...ref,
          id: UuidSchema.parse(randomUUID()),
          sourceChunkIndex: chunkIdx,
        });
      }

      for (const node of chunk.nodes) {
        const key = node.name.toLowerCase();
        const existing = nodesByName.get(key);
        const descriptor = chunk.corefByExtractedId.get(node.id);

        if (!existing) {
          nodesByName.set(key, node);
          chunkIndicesByExtractedId.set(node.id, new Set([chunkIdx]));
          if (descriptor) {
            corefByExtractedId.set(node.id, {
              identifyingDescription: descriptor.identifyingDescription,
              aliases: [...descriptor.aliases],
              referredToAsPronouns: [...descriptor.referredToAsPronouns],
            });
          }
        } else {
          chunkIndicesByExtractedId.get(existing.id)!.add(chunkIdx);
          if (descriptor) {
            const kept = corefByExtractedId.get(existing.id);
            if (kept) {
              for (const alias of descriptor.aliases) {
                if (!kept.aliases.includes(alias)) kept.aliases.push(alias);
              }
              for (const pronoun of descriptor.referredToAsPronouns) {
                if (!kept.referredToAsPronouns.includes(pronoun)) {
                  kept.referredToAsPronouns.push(pronoun);
                }
              }
            } else {
              corefByExtractedId.set(existing.id, {
                identifyingDescription: descriptor.identifyingDescription,
                aliases: [...descriptor.aliases],
                referredToAsPronouns: [...descriptor.referredToAsPronouns],
              });
            }
          }
        }
      }
    });
    const extractedNodes = [...nodesByName.values()];

    return {
      nodes: extractedNodes,
      chunkIndicesByExtractedId,
      corefByExtractedId,
      unresolvedReferences,
      metrics: {
        'episode.id': episode.id,
        'entityTypes.count': entityTypes ? Object.keys(entityTypes).length : 0,
        'chunks.count': chunks.length,
        'extracted.count': extractedNodes.length,
        'coref.unresolved.count': unresolvedReferences.length,
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
  ): Promise<{
    nodes: EntityNode[];
    corefByExtractedId: Map<Uuid, EntityCorefDescriptor>;
    unresolvedReferences: UnresolvedReference[];
  }> {
    const messages = buildExtractNodesMessages({
      episode,
      previousEpisodes,
      entityTypes,
      customInstructions,
    });
    const opts = {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'extract-nodes',
      tags: ['knowledge-graph', 'extraction.node'],
      validate: buildExtractNodesValidator({ entityTypes }),
    };

    const result = await invokeStructured(model, ExtractedEntitiesSchema, messages, opts);
    const unresolvedReferences = result.unresolvedReferences;

    const corefByExtractedId = new Map<Uuid, EntityCorefDescriptor>();
    const nodes: EntityNode[] = [];

    for (const e of result.extractedEntities) {
      const node = createEntityNode({
        name: e.name,
        graphId: episode.graphId,
        labels: resolveLabels(e.entityTypeId, entityTypes),
      });
      if (excludedEntityTypes?.length) {
        const specificLabel = node.labels.find((l) => l !== 'Entity') ?? 'Entity';
        if (excludedEntityTypes.includes(specificLabel)) continue;
      }
      nodes.push(node);
      corefByExtractedId.set(node.id, {
        identifyingDescription: e.identifyingDescription,
        aliases: e.aliases,
        referredToAsPronouns: e.referredToAsPronouns,
      });
    }
    return { nodes, corefByExtractedId, unresolvedReferences };
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
    const nodeNameById = new Map(nodes.map((n) => [n.id, n.name]));

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
          coreferences: selectCoreferencesForChunks(
            nodeCtx.committedCorefBindings,
            nodeNameById,
            nodeCtx.sourceChunkIndices,
          ),
          labelChunks: nodeCtx.chunks.length > 1,
        });
        const attrs = (await invokeStructured(model, entityType.schema, attrMessages, {
          callbacks: this.llmTracer.getCallbacks(ctx),
          runName: 'fill-entity-attributes',
          tags: ['knowledge-graph', 'attributes.entity'],
        })) as Record<string, unknown>;

        node.attributes = { ...node.attributes, ...attrs };
      });
    }
    await withConcurrency(this.kgConfig.memoryBackpressureConcurrencyLimit, tasks);

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
    const nodeNameById = new Map(nodes.map((n) => [n.id, n.name]));
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
            coreferences: selectCoreferencesForChunks(
              nodeContext.get(batchNodes[0].id)!.committedCorefBindings,
              nodeNameById,
              batchChunkIndices,
            ),
            labelChunks: chunks.length > 1,
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
    await withConcurrency(this.kgConfig.memoryBackpressureConcurrencyLimit, tasks);

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
