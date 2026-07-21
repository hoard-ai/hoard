import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import type { Uuid } from '@/common/schemas';
import { LlmService } from '@/llm/llm.service';
import { LLM_TRACER, NoOpLlmTracer } from '@/observability';
import { KG_TEST_GRAPH_ID, KG_TEST_USER_ID, u } from '@/test/factories';

import { EmbeddingService } from '../embedding';
import type { Community } from '../models';
import {
  type ClusterMatch,
  CommunityRepository,
  EntityEdgeRepository,
  EntityNodeRepository,
  type ExistingCommunitySnapshot,
} from '../repository/repositories';
import type { NodeName } from '../types';
import { summaryHash } from './community-utils';
import { CommunityService } from './community.service';

/**
 * Per-cluster routing decisions (drift, additions, removal, ambiguity) are
 * unit-tested in community-utils.spec.ts against `planRoutes`. This spec
 * verifies the end-to-end orchestration: cold-start full path, clean-match
 * short-circuit, collision resolution, and the per-entity update path.
 *
 * LLM call dispatch is keyed off the `runName` passed to the structured-
 * output runnable - stable per call site and independent of prompt or
 * schema details.
 */

function agg(a: Uuid, b: Uuid, edgeCount = 1): { a: Uuid; b: Uuid; edgeCount: number } {
  return a < b ? { a, b, edgeCount } : { a: b, b: a, edgeCount };
}

function makeCommunity(overrides: Partial<Community>): Community {
  return {
    id: u('default-community'),
    graphId: KG_TEST_GRAPH_ID,
    name: 'community' as NodeName,
    summary: '',
    nameEmbedding: null,
    memberIds: [],
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  };
}

function makeSnapshot(
  overrides: Partial<ExistingCommunitySnapshot>,
): ExistingCommunitySnapshot {
  return {
    id: u('default-existing'),
    name: 'existing-community' as NodeName,
    memberIds: [],
    summary: '',
    nameEmbedding: null,
    ...overrides,
  };
}

function hashesFor(
  memberIds: Uuid[],
  summaryById: Map<Uuid, string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const id of memberIds) out[id] = summaryHash(summaryById.get(id) ?? '');
  return out;
}

function sameMembers(a: Uuid[], b: Uuid[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((x, i) => x === sb[i]);
}

function humanText(messages: BaseMessage[]): string {
  const c = messages[1]?.content;
  return typeof c === 'string' ? c : '';
}

describe('CommunityService', () => {
  let service: CommunityService;

  let mockLlmService: DeepMocked<LlmService>;
  let mockEmbeddingService: DeepMocked<EmbeddingService>;
  let mockCommunityRepo: DeepMocked<CommunityRepository>;
  let mockEntityNodeRepo: DeepMocked<EntityNodeRepository>;
  let mockEntityEdgeRepo: DeepMocked<EntityEdgeRepository>;

  let mockModel: DeepMocked<BaseChatModel>;

  type ResolveResponse = {
    resolutions: Array<{ tempId: number; name: string }>;
  };
  type LlmHandlers = {
    summarizePair?: jest.Mock<Promise<{ summary: string }>, [BaseMessage[]]>;
    communityName?: jest.Mock<Promise<{ name: string }>, [BaseMessage[]]>;
    resolveCollisions?: jest.Mock<Promise<ResolveResponse>, [BaseMessage[]]>;
  };

  /**
   * Route the structured-output runnable's `invoke` to the right per-call-site
   * mock keyed by `runName`. Tests get back the mocks so they can assert call
   * counts and inspect the messages passed to each.
   */
  function setupStructuredOutput(handlers: LlmHandlers = {}): Required<LlmHandlers> {
    const summarizePair =
      handlers.summarizePair ??
      (jest.fn().mockResolvedValue({ summary: 'merged' }) as jest.Mock<
        Promise<{ summary: string }>,
        [BaseMessage[]]
      >);
    const communityName =
      handlers.communityName ??
      (jest.fn().mockResolvedValue({ name: 'noname' }) as jest.Mock<
        Promise<{ name: string }>,
        [BaseMessage[]]
      >);
    const resolveCollisions =
      handlers.resolveCollisions ??
      (jest.fn().mockImplementation(() => {
        throw new Error('unexpected resolver call');
      }) as jest.Mock<Promise<ResolveResponse>, [BaseMessage[]]>);

    const dispatch = (messages: BaseMessage[], opts?: { runName?: string }) => {
      switch (opts?.runName) {
        case 'community.summarize-pair':
          return summarizePair(messages);
        case 'community.community-name':
          return communityName(messages);
        case 'community.resolve-name-collisions':
          return resolveCollisions(messages);
        default:
          throw new Error(`unrecognized runName: ${String(opts?.runName)}`);
      }
    };

    mockModel.withStructuredOutput.mockReturnValue({ invoke: dispatch } as never);
    return { summarizePair, communityName, resolveCollisions };
  }

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CommunityService,
        { provide: LLM_TRACER, useValue: new NoOpLlmTracer() },
      ],
    })
      .useMocker(createMock)
      .compile();

    service = module.get(CommunityService);
    mockLlmService = module.get(LlmService);
    mockEmbeddingService = module.get(EmbeddingService);
    mockCommunityRepo = module.get(CommunityRepository);
    mockEntityNodeRepo = module.get(EntityNodeRepository);
    mockEntityEdgeRepo = module.get(EntityEdgeRepository);

    mockModel = createMock<BaseChatModel>();
    mockLlmService.getActiveModel.mockResolvedValue(mockModel);
    mockEmbeddingService.embedText.mockResolvedValue([0.1, 0.2, 0.3]);
    setupStructuredOutput();

    mockCommunityRepo.matchClusters.mockResolvedValue({
      matchesByClusterIndex: new Map(),
      existing: [],
    });
    mockCommunityRepo.deleteByGraphId.mockResolvedValue(undefined);
    mockCommunityRepo.deleteByIds.mockResolvedValue(undefined);
    mockCommunityRepo.saveBulk.mockResolvedValue(undefined);
    mockCommunityRepo.save.mockResolvedValue('' as never);
    mockCommunityRepo.applyIncrementalUpdate.mockResolvedValue(undefined);
    mockCommunityRepo.findByMemberId.mockResolvedValue(null);
    mockCommunityRepo.findByAnyMember.mockResolvedValue([]);
    mockCommunityRepo.findNamesByGraphId.mockResolvedValue([]);
    mockEntityNodeRepo.findSummariesByIds.mockResolvedValue([]);
    mockEntityNodeRepo.findIdsForGraph.mockResolvedValue([]);
    mockEntityEdgeRepo.findAggregatedNeighborCounts.mockResolvedValue([]);
    mockEntityEdgeRepo.findNeighborIds.mockResolvedValue([]);
  });

  afterEach(() => jest.clearAllMocks());

  describe('buildCommunities', () => {
    it('returns early and deletes existing communities when graph has too few nodes for a community', async () => {
      const llm = setupStructuredOutput();
      mockEntityNodeRepo.findIdsForGraph.mockResolvedValue([u('01'), u('02')]);

      await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

      expect(mockCommunityRepo.deleteByGraphId).toHaveBeenCalledWith(KG_TEST_GRAPH_ID);
      expect(mockCommunityRepo.saveBulk).not.toHaveBeenCalled();
      expect(llm.summarizePair).not.toHaveBeenCalled();
      expect(llm.communityName).not.toHaveBeenCalled();
      expect(llm.resolveCollisions).not.toHaveBeenCalled();
    });

    it('returns early when graph has no edges', async () => {
      const llm = setupStructuredOutput();
      mockEntityNodeRepo.findIdsForGraph.mockResolvedValue([u('01'), u('02'), u('03')]);
      mockEntityEdgeRepo.findAggregatedNeighborCounts.mockResolvedValue([]);

      await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

      expect(mockCommunityRepo.deleteByGraphId).toHaveBeenCalledWith(KG_TEST_GRAPH_ID);
      expect(llm.summarizePair).not.toHaveBeenCalled();
      expect(llm.communityName).not.toHaveBeenCalled();
    });

    it('drops clusters smaller than the minimum community size', async () => {
      // A triangle and a disconnected pair; only the triangle survives.
      const triangle: Uuid[] = [u('tri-a'), u('tri-b'), u('tri-c')];
      const pair: Uuid[] = [u('pair-a'), u('pair-b')];
      mockEntityNodeRepo.findIdsForGraph.mockResolvedValue([...triangle, ...pair].sort());
      mockEntityEdgeRepo.findAggregatedNeighborCounts.mockResolvedValue([
        agg(triangle[0], triangle[1], 1),
        agg(triangle[1], triangle[2], 1),
        agg(triangle[2], triangle[0], 1),
        agg(pair[0], pair[1], 1),
      ]);
      mockEntityNodeRepo.findSummariesByIds.mockImplementation((ids: Uuid[]) =>
        Promise.resolve(ids.map((id) => ({ id, summary: `about ${id}` }))),
      );
      setupStructuredOutput();

      await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

      expect(mockCommunityRepo.saveBulk).toHaveBeenCalledTimes(1);
      const saved = mockCommunityRepo.saveBulk.mock.calls[0][0];
      expect(saved).toHaveLength(1);
      expect(saved[0].memberIds).toEqual([...triangle].sort());
    });

    describe('with two communities', () => {
      // Two disconnected triangles - Louvain reliably produces two clusters.
      const cluster1: Uuid[] = [u('c1-a'), u('c1-b'), u('c1-c')];
      const cluster2: Uuid[] = [u('c2-a'), u('c2-b'), u('c2-c')];
      const summaryById = new Map<Uuid, string>([
        [cluster1[0], 'alice'],
        [cluster1[1], 'bob'],
        [cluster1[2], 'charlie'],
        [cluster2[0], 'dave'],
        [cluster2[1], 'eve'],
        [cluster2[2], 'frank'],
      ]);

      beforeEach(() => {
        mockEntityNodeRepo.findIdsForGraph.mockResolvedValue(
          [...cluster1, ...cluster2].sort(),
        );
        mockEntityEdgeRepo.findAggregatedNeighborCounts.mockResolvedValue([
          agg(cluster1[0], cluster1[1], 1),
          agg(cluster1[1], cluster1[2], 1),
          agg(cluster1[2], cluster1[0], 1),
          agg(cluster2[0], cluster2[1], 1),
          agg(cluster2[1], cluster2[2], 1),
          agg(cluster2[2], cluster2[0], 1),
        ]);
        mockEntityNodeRepo.findSummariesByIds.mockImplementation((ids: Uuid[]) =>
          Promise.resolve(
            ids
              .map((id) => ({ id, summary: summaryById.get(id) ?? '' }))
              .filter((r) => r.summary.length > 0),
          ),
        );
      });

      it('runs the full tournament for every cluster when none match existing (cold start)', async () => {
        mockCommunityRepo.matchClusters.mockResolvedValue({
          matchesByClusterIndex: new Map(),
          existing: [],
        });

        let namerCount = 0;
        const llm = setupStructuredOutput({
          communityName: jest.fn().mockImplementation(() => {
            namerCount += 1;
            return Promise.resolve({ name: `cluster-${namerCount}` });
          }),
        });

        await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

        expect(llm.resolveCollisions).not.toHaveBeenCalled();
        expect(mockCommunityRepo.saveBulk).toHaveBeenCalledTimes(1);
        const saved = mockCommunityRepo.saveBulk.mock.calls[0][0];
        expect(saved).toHaveLength(2);
        expect(saved.every((c) => c.memberIds.length === 3)).toBe(true);
        expect(saved.every((c) => c.nameEmbedding !== null)).toBe(true);
        const savedNames = saved.map((c) => c.name).sort();
        expect(savedNames).toEqual(['cluster-1', 'cluster-2']);
        expect(mockCommunityRepo.applyIncrementalUpdate).not.toHaveBeenCalled();
        expect(mockCommunityRepo.deleteByIds).not.toHaveBeenCalled();
      });

      it('routes colliding cluster names through the batched resolver', async () => {
        mockCommunityRepo.matchClusters.mockResolvedValue({
          matchesByClusterIndex: new Map(),
          existing: [],
        });

        const llm = setupStructuredOutput({
          communityName: jest.fn().mockResolvedValue({ name: 'duplicate-name' }),
          // Resolver renames each collider deterministically by tempId, so the
          // test doesn't need to know which route got which index.
          resolveCollisions: jest.fn().mockImplementation((messages: BaseMessage[]) => {
            const tempIds = [...humanText(messages).matchAll(/tempId:\s*(\d+)/g)].map(
              (m) => Number(m[1]),
            );
            return Promise.resolve({
              resolutions: tempIds.map((tempId) => ({
                tempId,
                name: `resolved-${tempId}`,
              })),
            });
          }),
        });

        await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

        expect(llm.resolveCollisions).toHaveBeenCalledTimes(1);
        expect(mockCommunityRepo.saveBulk).toHaveBeenCalledTimes(1);
        const saved = mockCommunityRepo.saveBulk.mock.calls[0][0];
        expect(saved).toHaveLength(2);
        const savedNames = saved.map((c) => c.name);
        // Both fresh names were colliders; resolver rewrote both.
        expect(savedNames.every((n) => n.startsWith('resolved-'))).toBe(true);
        expect(new Set(savedNames).size).toBe(2);
      });

      it('renames fresh names that collide with surviving (clean-route) names', async () => {
        const existing1Id = u('existing-1');
        const existing2Id = u('existing-2');
        // Cluster 1 matches existing-1 with clean hashes (surviving "shared-name").
        // Cluster 2 is unmatched and the namer will produce "shared-name" too,
        // forcing the resolver to rename just cluster 2.
        mockCommunityRepo.matchClusters.mockImplementation((_graphId, clusters) => {
          const map = new Map<number, ClusterMatch>();
          clusters.forEach((c, i) => {
            if (sameMembers(c, cluster1)) {
              map.set(i, {
                kind: 'matched-set',
                communityId: existing1Id,
                storedHashes: hashesFor(cluster1, summaryById),
              });
            }
          });
          return Promise.resolve({
            matchesByClusterIndex: map,
            existing: [
              makeSnapshot({
                id: existing1Id,
                name: 'shared-name' as NodeName,
                memberIds: cluster1,
                summary: 's1',
              }),
              makeSnapshot({
                id: existing2Id,
                name: 'orphan' as NodeName,
                memberIds: [u('orphan')],
                summary: 's-orphan',
              }),
            ],
          });
        });

        const llm = setupStructuredOutput({
          communityName: jest.fn().mockResolvedValue({ name: 'shared-name' }),
          resolveCollisions: jest.fn().mockImplementation((messages: BaseMessage[]) => {
            const tempIds = [...humanText(messages).matchAll(/tempId:\s*(\d+)/g)].map(
              (m) => Number(m[1]),
            );
            return Promise.resolve({
              resolutions: tempIds.map((tempId) => ({
                tempId,
                name: 'shared-name (2)',
              })),
            });
          }),
        });

        await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

        // Cluster 1 stayed clean; only cluster 2 was newly created and renamed.
        expect(mockCommunityRepo.saveBulk).toHaveBeenCalledTimes(1);
        const saved = mockCommunityRepo.saveBulk.mock.calls[0][0];
        expect(saved).toHaveLength(1);
        expect(saved[0].name).toBe('shared-name (2)');

        // The orphan existing community was unmatched and not consumed: stale.
        expect(mockCommunityRepo.deleteByIds).toHaveBeenCalledWith([existing2Id]);
        expect(llm.resolveCollisions).toHaveBeenCalledTimes(1);
      });

      it('skips all LLM work when every matched-set cluster has clean hashes', async () => {
        const existing1 = u('existing-1');
        const existing2 = u('existing-2');
        mockCommunityRepo.matchClusters.mockImplementation((_graphId, clusters) => {
          const map = new Map<number, ClusterMatch>();
          clusters.forEach((c, i) => {
            if (sameMembers(c, cluster1)) {
              map.set(i, {
                kind: 'matched-set',
                communityId: existing1,
                storedHashes: hashesFor(cluster1, summaryById),
              });
            } else if (sameMembers(c, cluster2)) {
              map.set(i, {
                kind: 'matched-set',
                communityId: existing2,
                storedHashes: hashesFor(cluster2, summaryById),
              });
            }
          });
          return Promise.resolve({
            matchesByClusterIndex: map,
            existing: [
              makeSnapshot({ id: existing1, memberIds: cluster1, summary: 's1' }),
              makeSnapshot({ id: existing2, memberIds: cluster2, summary: 's2' }),
            ],
          });
        });
        const llm = setupStructuredOutput();

        await service.buildCommunities(KG_TEST_USER_ID, KG_TEST_GRAPH_ID);

        expect(llm.summarizePair).not.toHaveBeenCalled();
        expect(llm.communityName).not.toHaveBeenCalled();
        expect(llm.resolveCollisions).not.toHaveBeenCalled();
        expect(mockEmbeddingService.embedText).not.toHaveBeenCalled();
        expect(mockCommunityRepo.saveBulk).not.toHaveBeenCalled();
        expect(mockCommunityRepo.applyIncrementalUpdate).not.toHaveBeenCalled();
        expect(mockCommunityRepo.deleteByIds).not.toHaveBeenCalled();
      });
    });
  });

  describe('updateCommunityForEntity', () => {
    const entityId = u('new-entity');

    it('refreshes the community summary when the entity is already a member', async () => {
      const otherMember = u('other-member');
      const existing = makeCommunity({
        id: u('existing-community'),
        name: 'old name' as NodeName,
        summary: 'old summary',
        memberIds: [otherMember, entityId].sort(),
      });
      mockCommunityRepo.findByMemberId.mockResolvedValue(existing);
      mockCommunityRepo.findNamesByGraphId.mockResolvedValue([
        existing.name,
        'other community' as NodeName,
      ]);
      mockEntityNodeRepo.findSummariesByIds.mockImplementation((ids: Uuid[]) =>
        Promise.resolve(ids.map((id) => ({ id, summary: `summary-of-${id}` }))),
      );
      const llm = setupStructuredOutput({
        summarizePair: jest.fn().mockResolvedValue({ summary: 'refreshed' }),
        communityName: jest.fn().mockResolvedValue({ name: 'refreshed name' }),
      });
      mockEmbeddingService.embedText.mockResolvedValue([0.7, 0.8, 0.9]);

      await service.updateCommunityForEntity(KG_TEST_USER_ID, KG_TEST_GRAPH_ID, entityId);

      expect(mockEntityEdgeRepo.findNeighborIds).not.toHaveBeenCalled();
      expect(mockCommunityRepo.findByAnyMember).not.toHaveBeenCalled();

      expect(mockCommunityRepo.applyIncrementalUpdate).toHaveBeenCalledTimes(1);
      const args = mockCommunityRepo.applyIncrementalUpdate.mock.calls[0][0];
      expect(args.id).toBe(existing.id);
      expect(sameMembers(args.memberIds, existing.memberIds)).toBe(true);
      expect(args.summary).toBe('refreshed');
      expect(args.name).toBe('refreshed name');
      expect(args.nameEmbedding).toEqual([0.7, 0.8, 0.9]);
      expect(llm.summarizePair).toHaveBeenCalledTimes(1);
      expect(llm.communityName).toHaveBeenCalledTimes(1);
      expect(mockEmbeddingService.embedText).toHaveBeenCalledTimes(1);

      // The target's own (about-to-be-replaced) name is excluded from the
      // namer's avoid-set. Inspecting the human-message data section is fine -
      // we're not asserting on prompt prose, just on which names the service
      // chose to pass through.
      const namerHuman = humanText(llm.communityName.mock.calls[0][0]);
      expect(namerHuman).toContain('other community');
      expect(namerHuman).not.toContain('old name');
    });

    it('skips when the entity has no neighbors yet', async () => {
      mockCommunityRepo.findByMemberId.mockResolvedValue(null);
      mockEntityEdgeRepo.findNeighborIds.mockResolvedValue([]);

      await service.updateCommunityForEntity(KG_TEST_USER_ID, KG_TEST_GRAPH_ID, entityId);

      expect(mockCommunityRepo.findByAnyMember).not.toHaveBeenCalled();
      expect(mockCommunityRepo.applyIncrementalUpdate).not.toHaveBeenCalled();
    });

    it("skips when none of the entity's neighbors are in any community", async () => {
      mockCommunityRepo.findByMemberId.mockResolvedValue(null);
      mockEntityEdgeRepo.findNeighborIds.mockResolvedValue([u('n1'), u('n2')]);
      mockCommunityRepo.findByAnyMember.mockResolvedValue([]);

      await service.updateCommunityForEntity(KG_TEST_USER_ID, KG_TEST_GRAPH_ID, entityId);

      expect(mockLlmService.getActiveModel).not.toHaveBeenCalled();
      expect(mockCommunityRepo.applyIncrementalUpdate).not.toHaveBeenCalled();
    });

    it('joins the mode-community of its neighbors and regenerates name/summary/embedding', async () => {
      const n1 = u('n1');
      const n2 = u('n2');
      const n3 = u('n3');
      const community = makeCommunity({
        id: u('target-community'),
        summary: 'old summary',
        memberIds: [n1, n2],
      });
      const otherCommunity = makeCommunity({
        id: u('other-community'),
        memberIds: [n3],
      });
      mockCommunityRepo.findByMemberId.mockResolvedValue(null);
      mockEntityEdgeRepo.findNeighborIds.mockResolvedValue([n1, n2, n3]);
      mockCommunityRepo.findByAnyMember.mockResolvedValue([community, otherCommunity]);
      mockEntityNodeRepo.findSummariesByIds.mockImplementation((ids: Uuid[]) =>
        Promise.resolve(ids.map((id) => ({ id, summary: `summary-of-${id}` }))),
      );
      const llm = setupStructuredOutput({
        summarizePair: jest.fn().mockResolvedValue({ summary: 'refreshed' }),
        communityName: jest.fn().mockResolvedValue({ name: 'fresh name' }),
      });
      mockEmbeddingService.embedText.mockResolvedValue([0.4, 0.5, 0.6]);

      await service.updateCommunityForEntity(KG_TEST_USER_ID, KG_TEST_GRAPH_ID, entityId);

      expect(mockCommunityRepo.applyIncrementalUpdate).toHaveBeenCalledTimes(1);
      const args = mockCommunityRepo.applyIncrementalUpdate.mock.calls[0][0];
      expect(args.id).toBe(community.id);
      expect(sameMembers(args.memberIds, [n1, n2, entityId])).toBe(true);
      expect(args.summary).toBe('refreshed');
      expect(args.name).toBe('fresh name');
      expect(args.nameEmbedding).toEqual([0.4, 0.5, 0.6]);
      expect(llm.summarizePair).toHaveBeenCalledTimes(1);
      expect(llm.communityName).toHaveBeenCalledTimes(1);
      expect(mockEmbeddingService.embedText).toHaveBeenCalledTimes(1);
    });
  });
});
