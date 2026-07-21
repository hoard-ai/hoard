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

import type { EntityCorefDescriptor } from '../extraction/types';
import { EntityNode, type EpisodicNode } from '../models';
import {
  buildCanonicalizeNodesMessages,
  buildCanonicalizeNodesValidator,
  buildDedupeNodesMessages,
  buildDedupeNodesValidator,
  CanonicalizeNodesSchema,
  NodeResolutionsSchema,
} from '../prompts';
import { selectChunkText } from '../prompts/text-utils';
import { EntityNodeRepository } from '../repository/repositories';
import { SearchBySimilarityParamsSchema, SearchByTextParamsSchema } from '../types';
import {
  CANDIDATE_LIMIT,
  COSINE_SIMILARITY_THRESHOLD,
  cosineSimilarity,
  LOW_ENTROPY_THRESHOLD,
  MAX_CANDIDATES,
  normalizeNameForEntropy,
  normalizeString,
  shannonEntropy,
} from './resolution-utils';
import { NodeResolutionResult } from './types';

@Injectable()
export class NodeResolutionService {
  constructor(
    private readonly entityNodeRepository: EntityNodeRepository,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  async collectCandidates(nodes: EntityNode[], graphId: Uuid): Promise<EntityNode[]> {
    const { candidates } = await this.collectCandidatesImpl(nodes, graphId);
    return candidates;
  }

  @Span('collectNodeCandidates', {
    observationKind: 'retriever',
    onResult: metricsOnResult,
  })
  private async collectCandidatesImpl(
    nodes: EntityNode[],
    graphId: Uuid,
  ): Promise<{ candidates: EntityNode[]; metrics: SpanMetrics }> {
    const results = await Promise.all(
      nodes.flatMap((n) => [
        this.entityNodeRepository.searchByName(
          SearchByTextParamsSchema.parse({
            query: n.name,
            graphIds: [graphId],
            limit: CANDIDATE_LIMIT,
          }),
        ),
        n.nameEmbedding !== null
          ? this.entityNodeRepository.searchBySimilarity(
              SearchBySimilarityParamsSchema.parse({
                embedding: n.nameEmbedding,
                graphIds: [graphId],
                limit: CANDIDATE_LIMIT,
              }),
            )
          : Promise.resolve([] as EntityNode[]),
      ]),
    );
    const seen = new Set<Uuid>();
    const candidates = results.flat().filter((n) => {
      if (seen.has(n.id)) return false;
      seen.add(n.id);
      return true;
    });
    return {
      candidates,
      metrics: {
        'input.count': nodes.length,
        'graph.id': graphId,
        'candidates.count': candidates.length,
      },
    };
  }

  async resolveNodes(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    chunkIndicesByExtractedId: Map<Uuid, Set<number>>,
    extractedNodes: EntityNode[],
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<NodeResolutionResult> {
    const { metrics: _m, ...rest } = await this.resolveNodesImpl(
      model,
      episode,
      chunks,
      chunkIndicesByExtractedId,
      extractedNodes,
      previousEpisodes,
      customInstructions,
      ctx,
    );
    return rest;
  }

  @Span('nodeResolution', { onResult: metricsOnResult })
  private async resolveNodesImpl(
    model: BaseChatModel,
    episode: EpisodicNode,
    chunks: string[],
    chunkIndicesByExtractedId: Map<Uuid, Set<number>>,
    extractedNodes: EntityNode[],
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<NodeResolutionResult & { metrics: SpanMetrics }> {
    const preexistingNodes = extractedNodes.length
      ? await this.collectCandidates(extractedNodes, episode.graphId)
      : [];

    const idMap = new Map<Uuid, Uuid>();
    const nodesMatchedToPreexistingNodes: Array<{
      extractedId: Uuid;
      preexistingNodeId: Uuid;
    }> = [];
    const llmCandidates = new Map<Uuid, EntityNode[]>();

    for (const extracted of extractedNodes) {
      const normalizedName = normalizeString(extracted.name);

      // Exact match check
      const exactMatch = preexistingNodes.find(
        (n) => normalizeString(n.name) === normalizedName,
      );
      if (exactMatch) {
        idMap.set(extracted.id, exactMatch.id);
        nodesMatchedToPreexistingNodes.push({
          extractedId: extracted.id,
          preexistingNodeId: exactMatch.id,
        });
        continue;
      }

      // Low entropy → skip cosine, go to LLM with all preexisting as candidates.
      // Mirrors Python: _normalize_name_for_fuzzy strips to [a-z0-9' ] (no spaces)
      // and _name_entropy computes entropy over that form.
      if (
        shannonEntropy(normalizeNameForEntropy(normalizedName)) < LOW_ENTROPY_THRESHOLD &&
        preexistingNodes.length > 0
      ) {
        llmCandidates.set(extracted.id, preexistingNodes);
        continue;
      }

      // Cosine similarity scan
      const embeddingCandidates = preexistingNodes.filter(
        (n) => n.nameEmbedding !== null,
      );

      if (extracted.nameEmbedding !== null && embeddingCandidates.length > 0) {
        const scored = embeddingCandidates
          .map((n) => ({
            node: n,
            score: cosineSimilarity(extracted.nameEmbedding!, n.nameEmbedding!),
          }))
          .filter((s) => s.score >= COSINE_SIMILARITY_THRESHOLD)
          .sort((a, b) => b.score - a.score)
          .slice(0, MAX_CANDIDATES);

        if (scored.length >= 1) {
          llmCandidates.set(
            extracted.id,
            scored.map((s) => s.node),
          );
          continue;
        }
        // 0 matches → new node, falls through
      }
    }

    // Batch LLM call for all ambiguous nodes
    if (llmCandidates.size > 0) {
      const llmExtractedWithIdx = extractedNodes
        .filter((n) => llmCandidates.has(n.id))
        .map((n, idx) => ({
          id: idx,
          name: n.name,
          labels: n.labels,
          entityId: n.id,
        }));

      const idxToEntityId = new Map(llmExtractedWithIdx.map((e) => [e.id, e.entityId]));

      // Collect unique candidate nodes across all batches, assigning a stable
      // integer candidateId so the LLM can reference them unambiguously.
      // String-name references are hallucination-prone; integer ids cannot be
      // invented (the LLM either picks one we sent or -1 for "no match").
      const candidateSet = new Map<Uuid, EntityNode>();
      for (const candidates of llmCandidates.values()) {
        for (const c of candidates) {
          candidateSet.set(c.id, c);
        }
      }
      const candidatesList = Array.from(candidateSet.values());
      const candidateIdToEntity = new Map<number, EntityNode>(
        candidatesList.map((n, idx) => [idx, n]),
      );
      const allCandidates = candidatesList.map((n, idx) => ({
        candidateId: idx,
        name: n.name,
        labels: n.labels,
      }));

      const extractedForPrompt = llmExtractedWithIdx.map(({ id, name, labels }) => ({
        id,
        name,
        labels,
      }));

      // Episode text = the chunks the resolved nodes came from (union across the
      // batch). Every extracted node carries originating chunk indices, so a miss
      // is a bookkeeping bug (mirrors the throw in nodeContext / summarizeNodes).
      const batchChunkIndices = new Set<number>();
      for (const { entityId } of llmExtractedWithIdx) {
        const indices = chunkIndicesByExtractedId.get(entityId);
        if (!indices) {
          throw new Error(
            `resolveNodes: extracted node ${entityId} has no originating chunk indices`,
          );
        }
        for (const idx of indices) batchChunkIndices.add(idx);
      }
      const episodeText = selectChunkText(batchChunkIndices, chunks);

      const messages = buildDedupeNodesMessages({
        episode: { ...episode, content: episodeText },
        previousEpisodes,
        extractedNodes: extractedForPrompt,
        candidateNodes: allCandidates,
        customInstructions,
      });

      const raw = await invokeStructured(model, NodeResolutionsSchema, messages, {
        callbacks: this.llmTracer.getCallbacks(ctx),
        runName: 'resolve-nodes',
        tags: ['knowledge-graph', 'resolution.node'],
        validate: buildDedupeNodesValidator({
          extractedNodes: extractedForPrompt,
          candidateNodes: allCandidates,
        }),
      });

      const resolutions = raw.entityResolutions;

      for (const resolution of resolutions) {
        const extractedId = idxToEntityId.get(resolution.id)!;

        if (resolution.duplicateCandidateId >= 0) {
          const canonical = candidateIdToEntity.get(resolution.duplicateCandidateId)!;
          idMap.set(extractedId, canonical.id);
          nodesMatchedToPreexistingNodes.push({
            extractedId,
            preexistingNodeId: canonical.id,
          });
          continue;
        }

        // nameEmbedding is cleared because it was computed for the old name -
        // persisting a stale embedding would corrupt vector search results.
        const node = extractedNodes.find((n) => n.id === extractedId);
        if (node && resolution.name !== node.name) {
          node.name = resolution.name;
          node.nameEmbedding = null;
        }
      }
    }

    const newNodes = extractedNodes.filter((n) => !idMap.has(n.id));

    return {
      newNodes,
      nodesMatchedToPreexistingNodes,
      preexistingCandidates: preexistingNodes,
      metrics: {
        'episode.id': episode.id,
        'extracted.count': extractedNodes.length,
        'preexisting.count': preexistingNodes.length,
        'new.count': newNodes.length,
        'matchedToPreexisting.count': nodesMatchedToPreexistingNodes.length,
      },
    };
  }

  /**
   * Within-episode canonicalization: one LLM pass over the episode's canonical
   * nodes with their coref descriptors, grouping entries that denote the same
   * real-world entity. Chunked extraction emits role nominals and short forms
   * as separate nodes, and no other stage can merge them: `resolveNodes` only
   * compares against the live graph and `dedupeAcrossBatch` is name-heuristic
   * only. The LLM group puts the canonical FIRST; each returned pair flips it
   * into `buildDirectedIdMap`'s (alias -> canonical) order.
   */
  async canonicalizeEpisodeNodes(
    model: BaseChatModel,
    nodes: EntityNode[],
    descriptors: Map<Uuid, EntityCorefDescriptor>,
    ctx?: LlmContext,
  ): Promise<[mergedAwayId: Uuid, keptId: Uuid][]> {
    const { pairs } = await this.canonicalizeEpisodeNodesImpl(
      model,
      nodes,
      descriptors,
      ctx,
    );
    return pairs;
  }

  @Span('canonicalizeEpisodeNodes', { onResult: metricsOnResult })
  private async canonicalizeEpisodeNodesImpl(
    model: BaseChatModel,
    nodes: EntityNode[],
    descriptors: Map<Uuid, EntityCorefDescriptor>,
    ctx?: LlmContext,
  ): Promise<{ pairs: [mergedAwayId: Uuid, keptId: Uuid][]; metrics: SpanMetrics }> {
    if (nodes.length < 2) {
      return { pairs: [], metrics: { 'nodes.count': nodes.length, 'pairs.found': 0 } };
    }

    const entities = nodes.map((n, id) => {
      const descriptor = descriptors.get(n.id);
      if (!descriptor) {
        throw new Error(
          `canonicalizeEpisodeNodes: node ${n.id} is missing its coref descriptor`,
        );
      }
      return {
        id,
        name: n.name,
        labels: n.labels,
        identifyingDescription: descriptor.identifyingDescription,
        aliases: descriptor.aliases,
        referredToAsPronouns: descriptor.referredToAsPronouns,
      };
    });

    const messages = buildCanonicalizeNodesMessages({ entities });
    const result = await invokeStructured(model, CanonicalizeNodesSchema, messages, {
      callbacks: this.llmTracer.getCallbacks(ctx),
      runName: 'canonicalize-nodes',
      tags: ['knowledge-graph', 'resolution.canonicalize'],
      validate: buildCanonicalizeNodesValidator({ entities }),
    });

    const pairs: [mergedAwayId: Uuid, keptId: Uuid][] = [];
    for (const group of result.duplicateGroups) {
      const keptId = nodes[group[0]].id;
      for (const idx of group.slice(1)) {
        pairs.push([nodes[idx].id, keptId]);
      }
    }

    return {
      pairs,
      metrics: {
        'nodes.count': nodes.length,
        'groups.found': result.duplicateGroups.length,
        'pairs.found': pairs.length,
      },
    };
  }

  // Within-batch dedup. The canonical pool is seeded with matched-preexisting
  // nodes from `resolveNodes` so a new node Y in one episode can be collapsed
  // onto preexisting X even when X wasn't in Y's own candidate set (it was
  // surfaced only by another episode's search). Without this, Y would silently
  // persist as a duplicate row alongside X. Mirrors upstream `dedupe_nodes_bulk`
  // (bulk_utils.py:414). New-vs-new keeps first-seen as canonical.
  dedupeAcrossBatch(
    newNodes: EntityNode[],
    matchedPreexistingNodes: EntityNode[],
  ): [aliasId: Uuid, canonicalId: Uuid][] {
    const { pairs } = this.dedupeAcrossBatchImpl(newNodes, matchedPreexistingNodes);
    return pairs;
  }

  @Span('dedupeNodesAcrossBatch', { onResult: metricsOnResult })
  private dedupeAcrossBatchImpl(
    newNodes: EntityNode[],
    matchedPreexistingNodes: EntityNode[],
  ): { pairs: [aliasId: Uuid, canonicalId: Uuid][]; metrics: SpanMetrics } {
    const isDuplicateNode = (a: EntityNode, b: EntityNode): boolean => {
      if (normalizeString(a.name) === normalizeString(b.name)) return true;
      return (
        a.nameEmbedding !== null &&
        b.nameEmbedding !== null &&
        cosineSimilarity(a.nameEmbedding, b.nameEmbedding) >= COSINE_SIMILARITY_THRESHOLD
      );
    };

    const pairs: [aliasId: Uuid, canonicalId: Uuid][] = [];
    const canonicalPool: EntityNode[] = [...matchedPreexistingNodes];
    for (const newNode of newNodes) {
      const match = canonicalPool.find((c) => isDuplicateNode(newNode, c));
      if (match) {
        pairs.push([newNode.id, match.id]);
      } else {
        canonicalPool.push(newNode);
      }
    }

    return {
      pairs,
      metrics: {
        'new.count': newNodes.length,
        'matched.count': matchedPreexistingNodes.length,
        'pairs.found': pairs.length,
      },
    };
  }
}
