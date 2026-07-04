import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

import type { Uuid } from '@/common/schemas';
import { invokeStructured } from '@/llm';
import type { LlmContext, LlmTracer } from '@/observability';

import type { EntityNodeRepository } from '../repository/repositories';
import {
  CrossEncoderScoreSchema,
  DEFAULT_MMR_LAMBDA,
  RRF_K,
  RRF_SECOND_RANK_BONUS,
  RRF_TOP_RANK_BONUS,
} from './types';

// ─── RRF ─────────────────────────────────────────────────────────────────────

/**
 * Reciprocal rank fusion across multiple ranked ID lists.
 * rank_const = 1 (matching the graphiti Python implementation).
 */
export function rrf<T extends string = Uuid>(
  resultIdLists: T[][],
  minScore = 0,
): [T[], number[]] {
  const scores = new Map<T, number>();

  for (const list of resultIdLists) {
    for (let i = 0; i < list.length; i++) {
      const id = list[i];
      scores.set(id, (scores.get(id) ?? 0) + 1 / (i + 1));
    }
  }

  const sorted = [...scores.entries()]
    .filter(([, score]) => score >= minScore)
    .sort((a, b) => b[1] - a[1]);

  return [sorted.map(([id]) => id), sorted.map(([, score]) => score)];
}

// ─── Weighted RRF ─────────────────────────────────────────────────────────────

/**
 * Weighted reciprocal rank fusion (qmd-style) for fusing many ranked lists from
 * different signals/sub-queries: score = Σ weight / (RRF_K + rank + 1), plus a
 * top-rank bonus so documents that rank #1 in any list aren't diluted by noisy
 * expansions. `weights` is parallel to `resultIdLists` (defaults to 1 each).
 */
export function weightedRrf<T extends string = Uuid>(
  resultIdLists: T[][],
  weights: number[] = [],
  minScore = 0,
): [T[], number[]] {
  const scores = new Map<T, number>();

  for (let listIdx = 0; listIdx < resultIdLists.length; listIdx++) {
    const list = resultIdLists[listIdx];
    const weight = weights[listIdx] ?? 1;
    for (let rank = 0; rank < list.length; rank++) {
      const id = list[rank];
      let contribution = weight / (RRF_K + rank + 1);
      if (rank === 0) contribution += RRF_TOP_RANK_BONUS;
      else if (rank <= 2) contribution += RRF_SECOND_RANK_BONUS;
      scores.set(id, (scores.get(id) ?? 0) + contribution);
    }
  }

  const sorted = [...scores.entries()]
    .filter(([, score]) => score >= minScore)
    .sort((a, b) => b[1] - a[1]);

  return [sorted.map(([id]) => id), sorted.map(([, score]) => score)];
}

// ─── MMR ─────────────────────────────────────────────────────────────────────

function l2Normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  if (mag === 0) return v;
  return v.map((x) => x / mag);
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * Maximal marginal relevance reranker.
 *
 * Candidates are L2-normalized; the query vector is used as-is (matching Python).
 * mmr_score = lambda * dot(queryVec, normCandidate) + (lambda-1) * maxPairwiseSim
 */
export function mmr<T extends string = Uuid>(
  queryVector: number[],
  idVectorPairs: Map<T, number[]>,
  lambda = DEFAULT_MMR_LAMBDA,
  minScore = -2.0,
): [T[], number[]] {
  if (idVectorPairs.size === 0) return [[], []];

  const ids = [...idVectorPairs.keys()];
  const normalized = ids.map((id) => l2Normalize(idVectorPairs.get(id)!));

  // Build full pairwise similarity matrix
  const simMatrix: number[][] = Array.from({ length: ids.length }, () =>
    Array.from<number>({ length: ids.length }).fill(0),
  );
  for (let i = 0; i < ids.length; i++) {
    for (let j = 0; j < i; j++) {
      const sim = dotProduct(normalized[i], normalized[j]);
      simMatrix[i][j] = sim;
      simMatrix[j][i] = sim;
    }
  }

  const mmrScores = ids.map((_, i) => {
    const maxSim = simMatrix[i].reduce((m, v) => (v > m ? v : m), -Infinity);
    return lambda * dotProduct(queryVector, normalized[i]) + (lambda - 1) * maxSim;
  });

  const scored = ids
    .map((id, i) => [id, mmrScores[i]] as [T, number])
    .filter(([, score]) => score >= minScore)
    .sort((a, b) => b[1] - a[1]);

  return [scored.map(([id]) => id), scored.map(([, score]) => score)];
}

// ─── Node distance reranker ───────────────────────────────────────────────────

/**
 * Ranks nodes by their graph distance to a center node.
 * Nodes directly connected to the center get score=1; others get Infinity.
 * Returns 1/score so closer nodes rank higher.
 */
export async function nodeDistanceReranker(
  repo: EntityNodeRepository,
  nodeIds: Uuid[],
  centerNodeId: Uuid,
  minScore = 0,
): Promise<[Uuid[], number[]]> {
  const filteredIds = nodeIds.filter((id) => id !== centerNodeId);
  const scores = new Map<Uuid, number>();

  if (filteredIds.length > 0) {
    const results = await repo.getNodeDistanceScores(filteredIds, centerNodeId);
    for (const row of results) {
      scores.set(row.id, row.score);
    }
  }

  for (const id of filteredIds) {
    if (!scores.has(id)) scores.set(id, Infinity);
  }

  filteredIds.sort((a, b) => scores.get(a)! - scores.get(b)!);

  // Re-insert center node at front if it was in the original list
  const orderedIds = nodeIds.includes(centerNodeId)
    ? [centerNodeId, ...filteredIds]
    : filteredIds;

  if (nodeIds.includes(centerNodeId)) {
    scores.set(centerNodeId, 0.1);
  }

  const result = orderedIds
    .map((id) => {
      const s = scores.get(id) ?? Infinity;
      return [id, s === Infinity ? 0 : 1 / s] as [Uuid, number];
    })
    .filter(([, invScore]) => invScore >= minScore);

  return [result.map(([id]) => id), result.map(([, score]) => score)];
}

// ─── Episode mentions reranker ────────────────────────────────────────────────

/**
 * Reranks node IDs by the number of episodic mentions.
 * Uses RRF as a preliminary ranker, then sorts descending by mention count
 * so the most-mentioned nodes rank highest.
 */
export async function episodeMentionsReranker(
  repo: EntityNodeRepository,
  nodeIdLists: Uuid[][],
  minScore = 0,
): Promise<[Uuid[], number[]]> {
  const [sortedIds] = rrf<Uuid>(nodeIdLists);
  const scores = new Map<Uuid, number>();

  if (sortedIds.length > 0) {
    const results = await repo.getEpisodeMentionCounts(sortedIds);
    for (const row of results) {
      scores.set(row.id, row.score);
    }
  }

  sortedIds.sort((a, b) => scores.get(b)! - scores.get(a)!);

  const result = sortedIds
    .map((id) => [id, scores.get(id)!] as [Uuid, number])
    .filter(([, score]) => score >= minScore);

  return [result.map(([id]) => id), result.map(([, score]) => score)];
}

// ─── Cross-encoder reranker ───────────────────────────────────────────────────

// TODO: This LLM-based scorer is a placeholder. Python Graphiti uses a real
// neural cross-encoder model (graphiti_core/cross_encoder/) for higher-quality
// relevance scoring at lower latency. Replace with a dedicated cross-encoder
// inference endpoint when available.

/**
 * LLM-based cross-encoder reranker. Scores each item 0–100 for relevance to
 * the query and normalizes to [0, 1].
 */
export async function crossEncoderReranker(
  model: BaseChatModel,
  query: string,
  items: Array<{ id: Uuid; text: string }>,
  minScore = 0,
  opts?: { llmTracer?: LlmTracer; ctx?: LlmContext },
): Promise<[Uuid[], number[]]> {
  if (items.length === 0) return [[], []];

  const callbacks = opts?.llmTracer?.getCallbacks(opts.ctx) ?? [];

  const rawScores = await Promise.all(
    items.map((item) =>
      invokeStructured(
        model,
        CrossEncoderScoreSchema,
        [
          new SystemMessage(
            'Rate the relevance of the text to the query from 0 to 100. Respond with only a JSON object containing "score".',
          ),
          new HumanMessage(`Query: ${query}\n\nText: ${item.text}`),
        ],
        {
          callbacks,
          runName: 'cross-encoder-score',
          tags: ['knowledge-graph', 'rerank.cross-encoder'],
        },
      ),
    ),
  );

  const scored = items
    .map((item, i) => {
      const raw = rawScores[i];
      const normalized = (raw.score ?? 0) / 100;
      return [item.id, normalized] as [Uuid, number];
    })
    .filter(([, score]) => score >= minScore)
    .sort((a, b) => b[1] - a[1]);

  return [scored.map(([id]) => id), scored.map(([, score]) => score)];
}
