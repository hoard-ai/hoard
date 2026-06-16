import { createMock, DeepMocked } from '@golevelup/ts-jest';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Test, TestingModule } from '@nestjs/testing';

import { Uuid } from '@/common/schemas';
import { LLM_TRACER, NoOpLlmTracer } from '@/observability';
import {
  KG_DIFF_EMBEDDING,
  KG_HIGH_SIM_EMBEDDING,
  KG_NEAR_SAME_EMBEDDING,
  KG_TEST_GRAPH_ID,
  KgNodeFactory,
} from '@/test/factories';

import { EntityNode } from '../models';
import { EntityNodeRepository } from '../repository/repositories';
import { NodeResolutionService } from './node-resolution.service';

const u = (s: string) => s as Uuid;

const baseEpisode = KgNodeFactory.createEpisodicNode({
  name: 'Test Episode',
  content: 'Alice works at Acme Corp.',
  graphId: KG_TEST_GRAPH_ID,
});

function makeNode(name: string, embedding: number[] | null = null): EntityNode {
  return KgNodeFactory.createEntityNode({
    name,
    graphId: KG_TEST_GRAPH_ID,
    nameEmbedding: embedding,
  });
}

// Extracted nodes always carry originating chunk indices (single-chunk episode
// here); resolveNodes throws on the LLM path without them.
const chunkIndices = (...nodes: EntityNode[]): Map<Uuid, Set<number>> =>
  new Map(nodes.map((n) => [n.id, new Set([0])]));

describe('NodeResolutionService', () => {
  let service: NodeResolutionService;
  let mockModel: DeepMocked<BaseChatModel>;
  let mockRunnable: { invoke: jest.Mock };
  let mockNodeRepo: DeepMocked<EntityNodeRepository>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        NodeResolutionService,
        { provide: LLM_TRACER, useValue: new NoOpLlmTracer() },
      ],
    })
      .useMocker(createMock)
      .compile();

    service = module.get(NodeResolutionService);
    mockNodeRepo = module.get(EntityNodeRepository);

    mockNodeRepo.searchByName.mockResolvedValue([]);
    mockNodeRepo.searchBySimilarity.mockResolvedValue([]);

    mockModel = createMock<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  // ─── resolveNodes ──────────────────────────────────────────────────────────

  describe('resolveNodes', () => {
    it('should resolve exact name match without LLM call', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [makeNode('alice', KG_HIGH_SIM_EMBEDDING)]; // normalizes to same
      existing[0].id = u('existing-id');

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
      expect(result.idMap.get(extracted[0].id)).toBe('existing-id');
      expect(result.resolvedNodes).toHaveLength(0);
    });

    it('should add duplicate pair for exact name match', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [makeNode('alice', KG_HIGH_SIM_EMBEDDING)];
      existing[0].id = u('existing-id');

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(result.duplicatePairs).toHaveLength(1);
      expect(result.duplicatePairs[0]).toEqual({
        extractedId: extracted[0].id,
        canonicalId: u('existing-id'),
      });
    });

    it('should escalate single cosine candidate to LLM', async () => {
      const extracted = [makeNode('Alice Johnson', KG_HIGH_SIM_EMBEDDING)];
      const existing = [makeNode('Alice J.', KG_NEAR_SAME_EMBEDDING)];
      existing[0].id = u('cosine-id');

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice Johnson', duplicateCandidateId: 0 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
      expect(result.idMap.get(extracted[0].id)).toBe('cosine-id');
      expect(result.resolvedNodes).toHaveLength(0);
    });

    it('should add as new node when LLM rejects single cosine candidate', async () => {
      const extracted = [makeNode('Alice Johnson', KG_HIGH_SIM_EMBEDDING)];
      const existing = [makeNode('Alice J.', KG_NEAR_SAME_EMBEDDING)];
      existing[0].id = u('cosine-id');

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice Johnson', duplicateCandidateId: -1 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
      expect(result.idMap.has(extracted[0].id)).toBe(false);
      expect(result.resolvedNodes).toHaveLength(1);
    });

    it('should escalate multiple cosine candidates to LLM', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Alice Smith', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-1'),
        },
        {
          ...makeNode('Alice Jones', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-2'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice', duplicateCandidateId: 0 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
      expect(result.idMap.get(extracted[0].id)).toBe('exist-1');
    });

    it('should add duplicate pair when LLM returns a duplicate_name match', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Alice Smith', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-1'),
        },
        {
          ...makeNode('Alice Jones', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-2'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice', duplicateCandidateId: 0 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(result.duplicatePairs).toHaveLength(1);
      expect(result.duplicatePairs[0]).toEqual({
        extractedId: extracted[0].id,
        canonicalId: 'exist-1',
      });
    });

    it('should add node to resolvedNodes when LLM returns empty duplicate_name', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Alice Smith', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-1'),
        },
        {
          ...makeNode('Alice Jones', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-2'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice', duplicateCandidateId: -1 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(result.idMap.has(extracted[0].id)).toBe(false);
      expect(result.resolvedNodes).toContainEqual(
        expect.objectContaining({ id: extracted[0].id }),
      );
      expect(result.duplicatePairs).toHaveLength(0);
    });

    it('should map id when LLM returns duplicate_name matching an existing node', async () => {
      const extracted = [makeNode('Alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Alice Smith', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-1'),
        },
        {
          ...makeNode('Alice Jones', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-2'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice', duplicateCandidateId: 0 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(result.idMap.get(extracted[0].id)).toBe('exist-1');
    });

    it('should apply canonical name from LLM when different from extracted name', async () => {
      const extracted = [makeNode('alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Alice Smith', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-1'),
        },
        {
          ...makeNode('Alice Jones', KG_NEAR_SAME_EMBEDDING),
          id: u('exist-2'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'Alice Smith', duplicateCandidateId: -1 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(result.resolvedNodes[0].name).toBe('Alice Smith');
    });

    it('should bypass cosine for low-entropy names and go to LLM', async () => {
      // "bob" entropy ≈ 0.918 (b:2, o:1) - below the 1.5 threshold → skips cosine
      const extracted = [makeNode('bob', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('Bobby', KG_DIFF_EMBEDDING),
          id: u('bob-exist'),
        },
      ];

      mockRunnable.invoke.mockResolvedValue({
        entityResolutions: [{ id: 0, name: 'bob', duplicateCandidateId: 0 }],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
      expect(result.idMap.get(extracted[0].id)).toBe('bob-exist');
    });

    it('should use cosine for names with entropy above threshold (e.g. "alice")', async () => {
      // "alice" entropy ≈ 2.32 (a,l,i,c,e - all distinct) - above the 1.5 threshold → cosine path.
      // Existing node "alicia" does not exact-match "alice" after normalizeString, so the
      // cosine scan runs. With KG_DIFF_EMBEDDING the cosine score is below threshold, so
      // no candidate is found and alice is added as a new node without any LLM call.
      const extracted = [makeNode('alice', KG_HIGH_SIM_EMBEDDING)];
      const existing = [
        {
          ...makeNode('alicia', KG_DIFF_EMBEDDING),
          id: u('alicia-exist'),
        },
      ];

      jest.spyOn(service, 'collectCandidates').mockResolvedValue(existing);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
      expect(result.resolvedNodes).toHaveLength(1);
      expect(result.resolvedNodes[0].name).toBe('alice');
    });

    it('should return all as new nodes with empty idMap when no existing nodes', async () => {
      const extracted = [
        makeNode('Alice', KG_HIGH_SIM_EMBEDDING),
        makeNode('Bob', KG_HIGH_SIM_EMBEDDING),
      ];

      jest.spyOn(service, 'collectCandidates').mockResolvedValue([]);
      const result = await service.resolveNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        chunkIndices(...extracted),
        extracted,
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
      expect(result.resolvedNodes).toHaveLength(2);
      expect(result.idMap.size).toBe(0);
      expect(result.duplicatePairs).toHaveLength(0);
    });
  });

  // ─── collectCandidates ─────────────────────────────────────────────────────

  describe('collectCandidates', () => {
    it('returns deduped union of name-search and similarity-search results', async () => {
      const node = makeNode('Alice', KG_HIGH_SIM_EMBEDDING);
      const byName = { ...makeNode('Alice'), id: u('by-name') };
      const bySim = { ...makeNode('Alice'), id: u('by-sim') };
      const shared = { ...makeNode('Alice Common'), id: u('shared') };

      mockNodeRepo.searchByName.mockResolvedValue([byName, shared]);
      mockNodeRepo.searchBySimilarity.mockResolvedValue([bySim, shared]);

      const result = await service.collectCandidates([node], KG_TEST_GRAPH_ID);

      const ids = result.map((n) => n.id).sort();
      expect(ids).toEqual([u('by-name'), u('by-sim'), u('shared')].sort());
    });

    it('skips similarity search when nameEmbedding is null', async () => {
      const node = makeNode('Alice', null);

      await service.collectCandidates([node], KG_TEST_GRAPH_ID);

      expect(mockNodeRepo.searchByName).toHaveBeenCalled();
      expect(mockNodeRepo.searchBySimilarity).not.toHaveBeenCalled();
    });
  });

  // ─── dedupeAcrossBatch ─────────────────────────────────────────────────────

  describe('dedupeAcrossBatch', () => {
    it('two nodes with identical embeddings → second collapses onto first', () => {
      const a = { ...makeNode('Alice', [1, 0]), id: u('a') };
      const b = { ...makeNode('Alicia', [1, 0]), id: u('b') }; // cosine=1.0

      const pairs = service.dedupeAcrossBatch([a, b], []);

      expect(pairs).toEqual([[u('b'), u('a')]]);
    });

    it('orthogonal embeddings → no pairs (below cosine threshold)', () => {
      const a = { ...makeNode('Alice', [1, 0]), id: u('a') };
      const b = { ...makeNode('Bob', [0, 1]), id: u('b') };

      const pairs = service.dedupeAcrossBatch([a, b], []);

      expect(pairs).toEqual([]);
    });

    it('identical names with null embeddings → pair via exact name match', () => {
      const a = { ...makeNode('Alice', null), id: u('a') };
      const b = { ...makeNode('Alice', null), id: u('b') };

      const pairs = service.dedupeAcrossBatch([a, b], []);

      expect(pairs).toEqual([[u('b'), u('a')]]);
    });

    it('mixed null + embedded with same name → pair via exact name match', () => {
      const a = { ...makeNode('Alice', [1, 0]), id: u('a') };
      const b = { ...makeNode('Alice', null), id: u('b') };

      const pairs = service.dedupeAcrossBatch([a, b], []);

      expect(pairs).toEqual([[u('b'), u('a')]]);
    });

    it('different names with null embeddings → no pairs', () => {
      const a = { ...makeNode('Alice', null), id: u('a') };
      const b = { ...makeNode('Bob', null), id: u('b') };

      const pairs = service.dedupeAcrossBatch([a, b], []);

      expect(pairs).toEqual([]);
    });

    it('seeded canonical pool: new node collapses onto matched-existing', () => {
      const existing = { ...makeNode('Alice', [1, 0]), id: u('existing') };
      const newNode = { ...makeNode('Alicia', [1, 0]), id: u('new') };

      const pairs = service.dedupeAcrossBatch([newNode], [existing]);

      expect(pairs).toEqual([[u('new'), u('existing')]]);
    });
  });
});
