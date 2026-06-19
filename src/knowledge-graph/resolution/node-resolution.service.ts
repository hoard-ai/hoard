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

import { EntityNode, type EpisodicNode } from '../models';
import {
  buildDedupeNodesMessages,
  buildDedupeNodesValidator,
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
    chunkIndicesByNodeId: Map<Uuid, Set<number>>,
    extractedNodes: EntityNode[],
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<NodeResolutionResult> {
    const { metrics: _m, ...rest } = await this.resolveNodesImpl(
      model,
      episode,
      chunks,
      chunkIndicesByNodeId,
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
    chunkIndicesByNodeId: Map<Uuid, Set<number>>,
    extractedNodes: EntityNode[],
    previousEpisodes: EpisodicNode[] = [],
    customInstructions?: string,
    ctx?: LlmContext,
  ): Promise<NodeResolutionResult & { metrics: SpanMetrics }> {
    const existingNodes = extractedNodes.length
      ? await this.collectCandidates(extractedNodes, episode.graphId)
      : [];

    const idMap = new Map<Uuid, Uuid>();
    const duplicatePairs: Array<{
      extractedId: Uuid;
      canonicalId: Uuid;
    }> = [];
    const llmCandidates = new Map<Uuid, EntityNode[]>();

    for (const extracted of extractedNodes) {
      const normalizedName = normalizeString(extracted.name);

      // Exact match check
      const exactMatch = existingNodes.find(
        (n) => normalizeString(n.name) === normalizedName,
      );
      if (exactMatch) {
        idMap.set(extracted.id, exactMatch.id);
        duplicatePairs.push({
          extractedId: extracted.id,
          canonicalId: exactMatch.id,
        });
        continue;
      }

      // Low entropy → skip cosine, go to LLM with all existing as candidates.
      // Mirrors Python: _normalize_name_for_fuzzy strips to [a-z0-9' ] (no spaces)
      // and _name_entropy computes entropy over that form.
      if (
        shannonEntropy(normalizeNameForEntropy(normalizedName)) < LOW_ENTROPY_THRESHOLD &&
        existingNodes.length > 0
      ) {
        llmCandidates.set(extracted.id, existingNodes);
        continue;
      }

      // Cosine similarity scan
      const embeddingCandidates = existingNodes.filter((n) => n.nameEmbedding !== null);

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
        const indices = chunkIndicesByNodeId.get(entityId);
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
          duplicatePairs.push({ extractedId, canonicalId: canonical.id });
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

    const resolvedNodes = extractedNodes.filter((n) => !idMap.has(n.id));

    return {
      resolvedNodes,
      idMap,
      duplicatePairs,
      candidates: existingNodes,
      metrics: {
        'episode.id': episode.id,
        'extracted.count': extractedNodes.length,
        'existing.count': existingNodes.length,
        'resolved.count': resolvedNodes.length,
        'duplicates.count': duplicatePairs.length,
      },
    };
  }

  // Within-batch dedup. The canonical pool is seeded with matched-existing
  // nodes from `resolveNodes` so a new node Y in one episode can be collapsed
  // onto existing X even when X wasn't in Y's own candidate set (it was
  // surfaced only by another episode's search). Without this, Y would silently
  // persist as a duplicate row alongside X. Mirrors upstream `dedupe_nodes_bulk`
  // (bulk_utils.py:414). New-vs-new keeps first-seen as canonical.
  dedupeAcrossBatch(
    newNodes: EntityNode[],
    matchedExistingNodes: EntityNode[],
  ): [Uuid, Uuid][] {
    const { pairs } = this.dedupeAcrossBatchImpl(newNodes, matchedExistingNodes);
    return pairs;
  }

  @Span('dedupeNodesAcrossBatch', { onResult: metricsOnResult })
  private dedupeAcrossBatchImpl(
    newNodes: EntityNode[],
    matchedExistingNodes: EntityNode[],
  ): { pairs: [Uuid, Uuid][]; metrics: SpanMetrics } {
    const isDuplicateNode = (a: EntityNode, b: EntityNode): boolean => {
      if (normalizeString(a.name) === normalizeString(b.name)) return true;
      return (
        a.nameEmbedding !== null &&
        b.nameEmbedding !== null &&
        cosineSimilarity(a.nameEmbedding, b.nameEmbedding) >= COSINE_SIMILARITY_THRESHOLD
      );
    };

    const pairs: [Uuid, Uuid][] = [];
    const canonicalPool: EntityNode[] = [...matchedExistingNodes];
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
        'matched.count': matchedExistingNodes.length,
        'pairs.found': pairs.length,
      },
    };
  }
}
