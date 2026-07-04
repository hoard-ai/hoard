import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { mockDeep } from 'jest-mock-extended';

import type { Uuid } from '@/common/schemas';

import type { EntityNodeRepository } from '../repository/repositories';
import {
  crossEncoderReranker,
  episodeMentionsReranker,
  mmr,
  nodeDistanceReranker,
  rrf,
  weightedRrf,
} from './search-utils';
import { RRF_K, RRF_SECOND_RANK_BONUS, RRF_TOP_RANK_BONUS } from './types';

// Branded test sentinels - narrow string aliases for branded ids in mocks.
const u = (s: string) => s as Uuid;

// ─── rrf ─────────────────────────────────────────────────────────────────────

describe('rrf', () => {
  it('ranks a single list in input order with correct scores', () => {
    const [ids, scores] = rrf([['a', 'b', 'c']]);
    expect(ids).toEqual(['a', 'b', 'c']);
    expect(scores[0]).toBeCloseTo(1 / 1);
    expect(scores[1]).toBeCloseTo(1 / 2);
    expect(scores[2]).toBeCloseTo(1 / 3);
  });

  it('accumulates scores from two disjoint lists', () => {
    const [ids, scores] = rrf([['a'], ['b']]);
    // Both rank 0 in their respective lists → score = 1
    expect(scores[0]).toBeCloseTo(1);
    expect(scores[1]).toBeCloseTo(1);
    expect(ids).toHaveLength(2);
  });

  it('adds scores for an ID that appears in multiple lists', () => {
    // 'a' is rank 0 in both lists → 1/1 + 1/1 = 2
    // 'b' is rank 1 in list 1    → 1/2
    // 'c' is rank 1 in list 2    → 1/2
    const [ids, scores] = rrf([
      ['a', 'b'],
      ['a', 'c'],
    ]);
    expect(ids[0]).toBe('a');
    expect(scores[0]).toBeCloseTo(2);
    expect(scores[1]).toBeCloseTo(0.5);
    expect(scores[2]).toBeCloseTo(0.5);
  });

  it('filters IDs whose score is below minScore', () => {
    // 'a' score = 1/1 = 1, 'b' score = 1/2 = 0.5
    const [ids] = rrf([['a', 'b']], 0.6);
    expect(ids).toEqual(['a']);
  });

  it('returns empty arrays for an empty input', () => {
    expect(rrf([])).toEqual([[], []]);
  });

  it('ignores empty inner lists', () => {
    const [ids, scores] = rrf([[], ['a']]);
    expect(ids).toEqual(['a']);
    expect(scores[0]).toBeCloseTo(1);
  });
});

// ─── mmr ─────────────────────────────────────────────────────────────────────

describe('mmr', () => {
  it('returns empty arrays for empty candidates', () => {
    expect(mmr([1, 0], new Map())).toEqual([[], []]);
  });

  it('returns a single candidate with a computed score', () => {
    const [ids, scores] = mmr([1, 0], new Map([['a', [1, 0]]]), 1, -1);
    expect(ids).toEqual(['a']);
    expect(scores[0]).toBeCloseTo(1); // lambda=1: dot([1,0],[1,0]) = 1
  });

  it('with lambda=1 (pure relevance) ranks more similar candidate first', () => {
    // query = [1, 0]
    // 'a' = [1, 0] → cos = 1  (most similar)
    // 'b' = [0, 1] → cos = 0
    const [ids] = mmr(
      [1, 0],
      new Map([
        ['a', [1, 0]],
        ['b', [0, 1]],
      ]),
      1,
      -1,
    );
    expect(ids[0]).toBe('a');
  });

  it('with lambda=0 (pure diversity) ranks least correlated candidate first', () => {
    // query = [1, 0, 0]
    // 'a' = [1, 0, 0]  - strongly correlated with 'c'
    // 'b' = [0, 1, 0]  - uncorrelated with both others
    // 'c' = [0.9, 0.1, 0] - strongly correlated with 'a'
    // With lambda=0: mmr = -maxPairwiseSim
    // maxSim(b) ≈ 0.1  → mmr(b) ≈ -0.1  (highest / least penalised)
    // maxSim(a) ≈ 0.99 → mmr(a) ≈ -0.99 (most penalised)
    const [ids] = mmr(
      [1, 0, 0],
      new Map([
        ['a', [1, 0, 0]],
        ['b', [0, 1, 0]],
        ['c', [0.9, 0.1, 0]],
      ]),
      0,
      -1,
    );
    expect(ids[0]).toBe('b');
  });

  it('filters candidates below minScore', () => {
    // lambda=1, query=[1,0]: 'a' score≈1, 'b' score≈0
    const [ids] = mmr(
      [1, 0],
      new Map([
        ['a', [1, 0]],
        ['b', [0, 1]],
      ]),
      1,
      0.5,
    );
    expect(ids).toEqual(['a']);
  });

  it('handles a zero-vector candidate without producing NaN', () => {
    const [ids, scores] = mmr([1, 0], new Map([['a', [0, 0]]]), 1, -1);
    expect(ids).toEqual(['a']);
    expect(Number.isNaN(scores[0])).toBe(false);
  });
});

// ─── nodeDistanceReranker ─────────────────────────────────────────────────────

describe('nodeDistanceReranker', () => {
  let repo: ReturnType<typeof mockDeep<EntityNodeRepository>>;

  beforeEach(() => {
    repo = mockDeep<EntityNodeRepository>();
  });

  afterEach(() => jest.clearAllMocks());

  it('places a directly connected node above an unconnected node', async () => {
    // 'b' is connected (DB returns score=1), 'c' is not (absent from DB)
    repo.getNodeDistanceScores.mockResolvedValue([{ id: u('b'), score: 1 }]);

    const [ids, scores] = await nodeDistanceReranker(repo, [u('b'), u('c')], u('center'));
    expect(ids[0]).toBe('b');
    expect(scores[0]).toBeCloseTo(1); // 1/1
    expect(ids[1]).toBe('c');
    expect(scores[1]).toBeCloseTo(0); // 1/Infinity → 0
  });

  it('prepends the center node at rank 1 when it is in the input list', async () => {
    repo.getNodeDistanceScores.mockResolvedValue([{ id: u('b'), score: 1 }]);

    const [ids, scores] = await nodeDistanceReranker(
      repo,
      [u('center'), u('b'), u('c')],
      u('center'),
    );
    expect(ids[0]).toBe('center');
    expect(scores[0]).toBeCloseTo(10); // 1 / 0.1
  });

  it('does not prepend the center node when it is absent from the input list', async () => {
    repo.getNodeDistanceScores.mockResolvedValue([{ id: u('b'), score: 1 }]);

    const [ids] = await nodeDistanceReranker(repo, [u('b'), u('c')], u('center'));
    expect(ids[0]).toBe('b');
    expect(ids).not.toContain('center');
  });

  it('filters unconnected nodes when minScore > 0', async () => {
    repo.getNodeDistanceScores.mockResolvedValue([{ id: u('b'), score: 1 }]);

    // unconnected 'c' gets score 0, which is below minScore=0.5
    const [ids] = await nodeDistanceReranker(repo, [u('b'), u('c')], u('center'), 0.5);
    expect(ids).not.toContain('c');
    expect(ids).toContain('b');
  });

  it('returns empty and makes no DB call when nodeIds is empty', async () => {
    const [ids, scores] = await nodeDistanceReranker(repo, [], u('center'));
    expect(ids).toEqual([]);
    expect(scores).toEqual([]);
    expect(repo.getNodeDistanceScores).not.toHaveBeenCalled();
  });

  it('returns only the center node and makes no DB call when it is the only input', async () => {
    const [ids, scores] = await nodeDistanceReranker(repo, [u('center')], u('center'));
    expect(ids).toEqual(['center']);
    expect(scores[0]).toBeCloseTo(10); // 1 / 0.1
    expect(repo.getNodeDistanceScores).not.toHaveBeenCalled();
  });
});

// ─── episodeMentionsReranker ──────────────────────────────────────────────────

describe('episodeMentionsReranker', () => {
  let repo: ReturnType<typeof mockDeep<EntityNodeRepository>>;

  beforeEach(() => {
    repo = mockDeep<EntityNodeRepository>();
  });

  afterEach(() => jest.clearAllMocks());

  it('ranks the most-mentioned node first', async () => {
    repo.getEpisodeMentionCounts.mockResolvedValue([
      { id: u('node-a'), score: 20 },
      { id: u('node-b'), score: 5 },
    ]);

    const [ids, scores] = await episodeMentionsReranker(repo, [
      [u('node-a'), u('node-b')],
    ]);
    expect(ids[0]).toBe('node-a');
    expect(scores[0]).toBe(20);
    expect(ids[1]).toBe('node-b');
    expect(scores[1]).toBe(5);
  });

  it('places zero-mention nodes after positively-mentioned nodes', async () => {
    repo.getEpisodeMentionCounts.mockResolvedValue([
      { id: u('node-a'), score: 20 },
      { id: u('node-b'), score: 5 },
      { id: u('node-c'), score: 0 },
    ]);

    const [ids] = await episodeMentionsReranker(repo, [
      [u('node-a'), u('node-b'), u('node-c')],
    ]);
    expect(ids.indexOf(u('node-a'))).toBeLessThan(ids.indexOf(u('node-c')));
    expect(ids.indexOf(u('node-b'))).toBeLessThan(ids.indexOf(u('node-c')));
  });

  it('excludes zero-mention nodes when minScore = 1', async () => {
    repo.getEpisodeMentionCounts.mockResolvedValue([
      { id: u('node-a'), score: 20 },
      { id: u('node-b'), score: 5 },
      { id: u('node-c'), score: 0 },
    ]);

    const [ids] = await episodeMentionsReranker(
      repo,
      [[u('node-a'), u('node-b'), u('node-c')]],
      1,
    );
    expect(ids).not.toContain('node-c');
    expect(ids).toContain('node-a');
    expect(ids).toContain('node-b');
  });

  it('returns empty and makes no DB call for empty input', async () => {
    const [ids, scores] = await episodeMentionsReranker(repo, [[]]);
    expect(ids).toEqual([]);
    expect(scores).toEqual([]);
    expect(repo.getEpisodeMentionCounts).not.toHaveBeenCalled();
  });

  it('returns all nodes with score 0 when none are mentioned', async () => {
    repo.getEpisodeMentionCounts.mockResolvedValue([
      { id: u('node-a'), score: 0 },
      { id: u('node-b'), score: 0 },
    ]);

    const [ids, scores] = await episodeMentionsReranker(repo, [
      [u('node-a'), u('node-b')],
    ]);
    expect(ids).toHaveLength(2);
    expect(scores.every((s) => s === 0)).toBe(true);
  });
});

// ─── crossEncoderReranker ─────────────────────────────────────────────────────

describe('crossEncoderReranker', () => {
  let model: ReturnType<typeof mockDeep<BaseChatModel>>;
  let mockRunnable: { invoke: jest.Mock };

  beforeEach(() => {
    model = mockDeep<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    model.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  afterEach(() => jest.clearAllMocks());

  it('returns empty immediately for empty items without calling the model', async () => {
    const [ids, scores] = await crossEncoderReranker(model, 'query', []);
    expect(ids).toEqual([]);
    expect(scores).toEqual([]);
    expect(model.withStructuredOutput).not.toHaveBeenCalled();
  });

  it('normalises raw scores from [0, 100] to [0, 1]', async () => {
    mockRunnable.invoke.mockResolvedValue({ score: 80 });

    const [, scores] = await crossEncoderReranker(model, 'query', [
      { id: u('a'), text: 'foo' },
    ]);
    expect(scores[0]).toBeCloseTo(0.8);
  });

  it('ranks items by score descending', async () => {
    mockRunnable.invoke
      .mockResolvedValueOnce({ score: 40 }) // item 'a'
      .mockResolvedValueOnce({ score: 80 }); // item 'b'

    const [ids] = await crossEncoderReranker(model, 'query', [
      { id: u('a'), text: 'less relevant' },
      { id: u('b'), text: 'more relevant' },
    ]);
    expect(ids[0]).toBe('b');
    expect(ids[1]).toBe('a');
  });

  it('filters items whose normalised score is below minScore', async () => {
    mockRunnable.invoke
      .mockResolvedValueOnce({ score: 80 }) // 0.8 - passes
      .mockResolvedValueOnce({ score: 40 }); // 0.4 - filtered

    const [ids] = await crossEncoderReranker(
      model,
      'query',
      [
        { id: u('a'), text: 'high relevance' },
        { id: u('b'), text: 'low relevance' },
      ],
      0.5,
    );
    expect(ids).toEqual(['a']);
  });

  it('invokes the model once per item', async () => {
    mockRunnable.invoke.mockResolvedValue({ score: 50 });

    await crossEncoderReranker(model, 'query', [
      { id: u('a'), text: 'one' },
      { id: u('b'), text: 'two' },
      { id: u('c'), text: 'three' },
    ]);
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(3);
  });
});

// ─── weightedRrf ──────────────────────────────────────────────────────────────

describe('weightedRrf', () => {
  it('ranks a single list in order with the qmd-style score formula', () => {
    const [ids, scores] = weightedRrf([['a', 'b', 'c']]);
    expect(ids).toEqual(['a', 'b', 'c']);
    expect(scores[0]).toBeCloseTo(1 / (RRF_K + 1) + RRF_TOP_RANK_BONUS);
    expect(scores[1]).toBeCloseTo(1 / (RRF_K + 2) + RRF_SECOND_RANK_BONUS);
    expect(scores[2]).toBeCloseTo(1 / (RRF_K + 3) + RRF_SECOND_RANK_BONUS);
  });

  it('applies the per-list weight to the rank contribution', () => {
    const [, base] = weightedRrf([['a']], [1]);
    const [, doubled] = weightedRrf([['a']], [2]);
    // Both get the same top-rank bonus; the weighted rank term doubles.
    expect(doubled[0] - RRF_TOP_RANK_BONUS).toBeCloseTo(
      2 * (base[0] - RRF_TOP_RANK_BONUS),
    );
  });

  it('defaults missing weights to 1', () => {
    const [, withDefault] = weightedRrf([['a'], ['b']]);
    const [, explicit] = weightedRrf([['a'], ['b']], [1, 1]);
    expect(withDefault).toEqual(explicit);
  });

  it('accumulates contributions for an id present in multiple lists', () => {
    const [ids, scores] = weightedRrf([
      ['a', 'b'],
      ['a', 'c'],
    ]);
    expect(ids[0]).toBe('a');
    // 'a' is rank 0 in both lists, so it outscores the rank-1 entries.
    expect(scores[0]).toBeGreaterThan(scores[1]);
  });

  it('filters ids below minScore', () => {
    const [ids] = weightedRrf([['a', 'b']], [1, 1], 1);
    // No single entry reaches a score of 1 with k=60, so all are filtered.
    expect(ids).toEqual([]);
  });

  it('returns empty arrays for empty input', () => {
    expect(weightedRrf([])).toEqual([[], []]);
  });
});
