import { createMock, DeepMocked } from '@golevelup/ts-jest';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Test, TestingModule } from '@nestjs/testing';

import { Uuid, UuidSchema } from '@/common/schemas';
import { LlmService } from '@/llm/llm.service';
import { LLM_TRACER, NoOpLlmTracer } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';
import {
  KG_REFERENCE_TIME,
  KG_TEST_GRAPH_ID,
  KG_TEST_SAGA_ID,
  KG_TEST_USER_ID,
  KgEdgeFactory,
  KgNodeFactory,
  makeEpisode,
  u,
} from '@/test/factories';

import { CommunityMaintenanceService } from '../community';
import { EmbeddingService } from '../embedding';
import { EdgeExtractionService, NodeExtractionService } from '../extraction';
import { EntityEdge, EntityNode } from '../models';
import {
  EntityEdgeRepository,
  EntityNodeRepository,
  EpisodicEdgeRepository,
  EpisodicNodeRepository,
  SagaRepository,
} from '../repository/repositories';
import { EdgeResolutionService, NodeResolutionService } from '../resolution';
import { EpisodeService } from './episode.service';

// Extraction now returns items plus their chunk indices; chunks are irrelevant
// to these mocked tests, so the index maps are left empty.
const nodesResult = (nodes: EntityNode[]) => ({
  nodes,
  // Each extracted node carries originating chunk indices (single-chunk episode here).
  chunkIndicesByNodeId: new Map<Uuid, Set<number>>(
    nodes.map((n) => [n.id, new Set([0])]),
  ),
});
const edgesResult = (edges: EntityEdge[]) => ({
  edges,
  chunkIndicesByEdgeId: new Map<Uuid, Set<number>>(
    edges.map((e) => [e.id, new Set([0])]),
  ),
});

describe('EpisodeService', () => {
  let service: EpisodeService;

  let mockLlmService: DeepMocked<LlmService>;
  let mockCommunityMaintenance: DeepMocked<CommunityMaintenanceService>;
  let mockEmbeddingService: DeepMocked<EmbeddingService>;
  let mockNodeExtraction: DeepMocked<NodeExtractionService>;
  let mockEdgeExtraction: DeepMocked<EdgeExtractionService>;
  let mockNodeResolution: DeepMocked<NodeResolutionService>;
  let mockEdgeResolution: DeepMocked<EdgeResolutionService>;
  let mockEntityNodeRepo: DeepMocked<EntityNodeRepository>;
  let mockEntityEdgeRepo: DeepMocked<EntityEdgeRepository>;
  let mockEpisodicNodeRepo: DeepMocked<EpisodicNodeRepository>;
  let mockEpisodicEdgeRepo: DeepMocked<EpisodicEdgeRepository>;
  let mockSagaRepo: DeepMocked<SagaRepository>;
  let mockPrisma: DeepMocked<PrismaService>;

  let mockModel: DeepMocked<BaseChatModel>;
  let mockRunnable: { invoke: jest.Mock };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [EpisodeService, { provide: LLM_TRACER, useValue: new NoOpLlmTracer() }],
    })
      .useMocker(createMock)
      .compile();

    service = module.get(EpisodeService);
    mockLlmService = module.get(LlmService);
    mockCommunityMaintenance = module.get(CommunityMaintenanceService);
    mockEmbeddingService = module.get(EmbeddingService);
    mockNodeExtraction = module.get(NodeExtractionService);
    mockEdgeExtraction = module.get(EdgeExtractionService);
    mockNodeResolution = module.get(NodeResolutionService);
    mockEdgeResolution = module.get(EdgeResolutionService);
    mockEntityNodeRepo = module.get(EntityNodeRepository);
    mockEntityEdgeRepo = module.get(EntityEdgeRepository);
    mockEpisodicNodeRepo = module.get(EpisodicNodeRepository);
    mockEpisodicEdgeRepo = module.get(EpisodicEdgeRepository);
    mockSagaRepo = module.get(SagaRepository);
    mockPrisma = module.get(PrismaService);

    mockModel = createMock<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);

    // persistPhase wraps writes in prisma.$transaction; run the callback so the
    // repo mocks below are actually invoked (the dummy tx is ignored by them).
    mockPrisma.$transaction.mockImplementation((fn: (tx: unknown) => unknown) =>
      Promise.resolve(fn({})),
    );

    // Default mock implementations
    mockLlmService.getActiveModel.mockResolvedValue(mockModel);
    mockEpisodicNodeRepo.retrieveEpisodes.mockResolvedValue([]);
    mockEpisodicNodeRepo.saveBulk.mockResolvedValue(undefined);
    mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([]));
    mockNodeExtraction.fillEntityAttributes.mockResolvedValue(undefined);
    mockNodeExtraction.summarizeNodes.mockResolvedValue(undefined);
    mockNodeResolution.resolveNodes.mockResolvedValue({
      resolvedNodes: [],
      idMap: new Map(),
      duplicatePairs: [],
      candidates: [],
    });
    mockNodeResolution.dedupeAcrossBatch.mockReturnValue([]);
    mockEmbeddingService.embedNodes.mockResolvedValue([]);
    mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([]));
    mockEdgeExtraction.fillEdgeAttributes.mockResolvedValue(undefined);
    mockEdgeExtraction.extractEdgeTimestampsFallback.mockResolvedValue(undefined);
    mockEmbeddingService.embedEdges.mockResolvedValue([]);
    mockEdgeResolution.resolveEdges.mockResolvedValue({
      resolvedEdges: [],
      invalidatedEdges: [],
      newEdges: [],
    });
    // Default passthrough: dedup returns the flat distinct edge set (no merges).
    // Tests asserting cross-batch dedup behavior override this.
    mockEdgeResolution.dedupeAcrossBatch.mockImplementation((_m, edges) =>
      Promise.resolve(edges.flat()),
    );
    mockEpisodicEdgeRepo.saveBulk.mockResolvedValue(undefined);
    mockEntityNodeRepo.saveBulk.mockResolvedValue(undefined);
    mockEntityEdgeRepo.saveBulk.mockResolvedValue(undefined);
    mockCommunityMaintenance.scheduleMaintenance.mockResolvedValue(undefined);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  // ─── Pipeline orchestration: per-step behavior for a single-episode batch ───

  describe('addEpisodes - pipeline orchestration', () => {
    it('passes per-episode previous-episodes context to extractNodes', async () => {
      const prevEpisode = KgNodeFactory.createEpisodicNode({
        name: 'Prior',
        content: 'Alice works at Acme Corp.',
        validAt: KG_REFERENCE_TIME,
        graphId: KG_TEST_GRAPH_ID,
      });
      mockEpisodicNodeRepo.retrieveEpisodes.mockResolvedValue([prevEpisode]);

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockNodeExtraction.extractNodes).toHaveBeenCalledWith(
        mockModel,
        expect.objectContaining({ name: 'ep1', graphId: KG_TEST_GRAPH_ID }),
        expect.any(Array),
        [prevEpisode],
        undefined,
        undefined,
        undefined,
        expect.anything(),
      );
    });

    it('embeds extracted nodes in a single batched call', async () => {
      const nodeA = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const nodeB = KgNodeFactory.createEntityNode({ name: 'Bob' });
      mockNodeExtraction.extractNodes
        .mockResolvedValueOnce(nodesResult([nodeA]))
        .mockResolvedValueOnce(nodesResult([nodeB]));

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      expect(mockEmbeddingService.embedNodes).toHaveBeenCalledTimes(1);
      expect(mockEmbeddingService.embedNodes).toHaveBeenCalledWith([nodeA, nodeB]);
    });

    it('calls resolveNodes with embedded nodes', async () => {
      const extracted = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const embedded = { ...extracted, nameEmbedding: [1, 0, 0] };

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([extracted]));
      mockEmbeddingService.embedNodes.mockResolvedValue([embedded]);

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockNodeResolution.resolveNodes).toHaveBeenCalledWith(
        mockModel,
        expect.anything(),
        expect.any(Array),
        expect.any(Map),
        [embedded],
        [],
        undefined,
        expect.anything(),
      );
    });

    it('extracts edges with canonical nodes (resolved + matched existing)', async () => {
      const resolved = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const existing = {
        ...KgNodeFactory.createEntityNode({ name: 'Bob' }),
        id: u('existing-bob-id'),
      };
      const alias = KgNodeFactory.createEntityNode({ name: 'Robert' });

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([resolved, alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolved, nameEmbedding: null },
        { ...alias, nameEmbedding: null },
      ]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        resolvedNodes: [resolved],
        idMap: new Map([[alias.id, existing.id]]),
        duplicatePairs: [{ extractedId: alias.id, canonicalId: existing.id }],
        candidates: [existing],
      });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockEdgeExtraction.extractEdges).toHaveBeenCalledWith(
        mockModel,
        expect.anything(),
        expect.any(Array),
        expect.arrayContaining([resolved, existing]),
        [],
        KG_REFERENCE_TIME,
        undefined,
        undefined,
        undefined,
        expect.anything(),
      );
    });

    it('embeds extracted edges in a single batched call', async () => {
      const edgeA = KgEdgeFactory.createEntityEdge({
        name: 'WORKS_AT',
        sourceNodeId: u('s1'),
        targetNodeId: u('t1'),
        fact: 'fact 1',
      });
      const edgeB = KgEdgeFactory.createEntityEdge({
        name: 'KNOWS',
        sourceNodeId: u('s2'),
        targetNodeId: u('t2'),
        fact: 'fact 2',
      });
      mockEdgeExtraction.extractEdges
        .mockResolvedValueOnce(edgesResult([edgeA]))
        .mockResolvedValueOnce(edgesResult([edgeB]));

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      expect(mockEmbeddingService.embedEdges).toHaveBeenCalledTimes(1);
      expect(mockEmbeddingService.embedEdges).toHaveBeenCalledWith([edgeA, edgeB]);
    });

    it('calls resolveEdges with embedded edges and canonicalIdByNodeId', async () => {
      const edge = KgEdgeFactory.createEntityEdge({
        name: 'WORKS_AT',
        sourceNodeId: u('src'),
        targetNodeId: u('tgt'),
        fact: 'Alice works at Acme Corp',
      });
      const embeddedEdge = { ...edge, factEmbedding: [1, 0, 0] };

      mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([edge]));
      mockEmbeddingService.embedEdges.mockResolvedValue([embeddedEdge]);

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockEdgeResolution.resolveEdges).toHaveBeenCalledWith(
        mockModel,
        expect.any(Array), // episodes
        expect.any(Array), // chunksPerEpisode
        expect.any(Map), // chunkSources
        [embeddedEdge],
        expect.any(Map), // canonicalIdByNodeId
        KG_REFERENCE_TIME,
        [],
        undefined,
        expect.anything(),
      );
    });

    it('returns one result entry per input episode', async () => {
      const result = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2'), makeEpisode('ep3')],
      });

      expect(result).toHaveLength(3);
      result.forEach((entry, i) => {
        expect(entry.episode.name).toBe(`ep${i + 1}`);
        expect(entry.nodes).toBeInstanceOf(Array);
        expect(entry.edges).toBeInstanceOf(Array);
        expect(entry.invalidatedEdges).toBeInstanceOf(Array);
        expect(entry.episodicEdges).toBeInstanceOf(Array);
      });
    });

    it('builds one episodic edge per canonical node referenced by each episode', async () => {
      const resolved = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const alias = KgNodeFactory.createEntityNode({ name: 'Bobby' });
      const existing = {
        ...KgNodeFactory.createEntityNode({ name: 'Bob' }),
        id: u('bob-id'),
      };

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([resolved, alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolved, nameEmbedding: null },
        { ...alias, nameEmbedding: null },
      ]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        resolvedNodes: [resolved],
        idMap: new Map([[alias.id, existing.id]]),
        duplicatePairs: [{ extractedId: alias.id, canonicalId: existing.id }],
        candidates: [existing],
      });

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(entry.episodicEdges).toHaveLength(2);
      expect(entry.episodicEdges.map((e) => e.targetNodeId)).toEqual(
        expect.arrayContaining([resolved.id, existing.id]),
      );
    });
  });

  // ─── Pass-1: resolve nodes against the live graph ──────────────────────────

  describe('addEpisodes - pass-1 dedup (vs live graph)', () => {
    it('alias node is excluded from result entries when resolveNodes returns a duplicate pair', async () => {
      const canonical = KgNodeFactory.createEntityNode({
        name: 'Alice',
        graphId: KG_TEST_GRAPH_ID,
      });
      const alias = KgNodeFactory.createEntityNode({
        name: 'Alice Smith',
        graphId: KG_TEST_GRAPH_ID,
      });

      mockNodeExtraction.extractNodes
        .mockResolvedValueOnce(nodesResult([canonical]))
        .mockResolvedValueOnce(nodesResult([alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([canonical, alias]);
      mockNodeResolution.resolveNodes
        .mockResolvedValueOnce({
          resolvedNodes: [canonical],
          idMap: new Map([[canonical.id, canonical.id]]),
          duplicatePairs: [],
          candidates: [],
        })
        .mockResolvedValueOnce({
          resolvedNodes: [],
          duplicatePairs: [{ extractedId: alias.id, canonicalId: canonical.id }],
          idMap: new Map([[alias.id, canonical.id]]),
          candidates: [],
        });

      const result = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      const allNodes = result.flatMap((r) => r.nodes);
      expect(allNodes.find((n) => n.id === alias.id)).toBeUndefined();
      expect(allNodes.find((n) => n.id === canonical.id)).toBeDefined();
    });

    it('existing node referenced as canonical target is pulled into the matching episode entry', async () => {
      const existingCanonical = KgNodeFactory.createEntityNode({
        name: 'Alice',
        graphId: KG_TEST_GRAPH_ID,
      });
      const alias = KgNodeFactory.createEntityNode({
        name: 'Alice Smith',
        graphId: KG_TEST_GRAPH_ID,
      });

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([alias]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        resolvedNodes: [],
        duplicatePairs: [{ extractedId: alias.id, canonicalId: existingCanonical.id }],
        idMap: new Map([[alias.id, existingCanonical.id]]),
        candidates: [existingCanonical],
      });

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(entry.nodes.find((n) => n.id === existingCanonical.id)).toBeDefined();
      expect(entry.nodes.find((n) => n.id === alias.id)).toBeUndefined();
    });

    it('canonical extracted by two episodes is saved exactly once via saveBulk', async () => {
      const canonical = KgNodeFactory.createEntityNode({
        name: 'Alice',
        graphId: KG_TEST_GRAPH_ID,
      });
      const alias = KgNodeFactory.createEntityNode({
        name: 'Alice Smith',
        graphId: KG_TEST_GRAPH_ID,
      });

      mockNodeExtraction.extractNodes
        .mockResolvedValueOnce(nodesResult([canonical]))
        .mockResolvedValueOnce(nodesResult([alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([canonical, alias]);
      mockNodeResolution.resolveNodes
        .mockResolvedValueOnce({
          resolvedNodes: [canonical],
          idMap: new Map([[canonical.id, canonical.id]]),
          duplicatePairs: [],
          candidates: [],
        })
        .mockResolvedValueOnce({
          resolvedNodes: [canonical],
          duplicatePairs: [{ extractedId: alias.id, canonicalId: canonical.id }],
          idMap: new Map([[alias.id, canonical.id]]),
          candidates: [],
        });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      const savedNodes = mockEntityNodeRepo.saveBulk.mock.calls[0]?.[0];
      expect(savedNodes.filter((n) => n.id === canonical.id)).toHaveLength(1);
    });
  });

  // ─── Pass-2: within-batch dedup is delegated to NodeResolutionService ──────
  // Logic-level unit tests for the dedup itself live in
  // node-resolution.service.spec.ts ('dedupeAcrossBatch'). The orchestration
  // test below verifies that pairs returned by the service participate in the
  // final canonical projection.

  describe('addEpisodes - pass-2 dedup (orchestration)', () => {
    it('pairs returned by dedupeAcrossBatch are folded into canonicalIdByNodeId, collapsing the alias', async () => {
      const canonical = KgNodeFactory.createEntityNode({
        name: 'Alice',
        graphId: KG_TEST_GRAPH_ID,
      });
      const alias = KgNodeFactory.createEntityNode({
        name: 'Alicia',
        graphId: KG_TEST_GRAPH_ID,
      });

      mockNodeExtraction.extractNodes
        .mockResolvedValueOnce(nodesResult([canonical]))
        .mockResolvedValueOnce(nodesResult([alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([canonical, alias]);
      mockNodeResolution.resolveNodes
        .mockResolvedValueOnce({
          resolvedNodes: [canonical],
          idMap: new Map([[canonical.id, canonical.id]]),
          duplicatePairs: [],
          candidates: [],
        })
        .mockResolvedValueOnce({
          resolvedNodes: [alias],
          idMap: new Map([[alias.id, alias.id]]),
          duplicatePairs: [],
          candidates: [],
        });
      mockNodeResolution.dedupeAcrossBatch.mockReturnValue([[alias.id, canonical.id]]);

      const result = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      const allNodes = result.flatMap((r) => r.nodes);
      expect(allNodes.find((n) => n.id === canonical.id)).toBeDefined();
      expect(allNodes.find((n) => n.id === alias.id)).toBeUndefined();
    });
  });

  // ─── Saga handling per episode (sagaId) ─────────────────────────────────

  describe('addEpisodes - saga handling', () => {
    const savedEpisodes = () =>
      mockEpisodicNodeRepo.saveBulk.mock.calls.flatMap(([nodes]) => nodes);

    it('creates Saga and tags the episode with sagaId when provided', async () => {
      const ep = makeEpisode('ep1');
      ep.sagaId = KG_TEST_SAGA_ID;

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [ep],
      });

      expect(mockSagaRepo.createIfNotExists).toHaveBeenCalledWith(
        expect.objectContaining({
          id: KG_TEST_SAGA_ID,
          graphId: KG_TEST_GRAPH_ID,
        }),
        expect.anything(),
      );
      expect(savedEpisodes()).toEqual([
        expect.objectContaining({ sagaId: KG_TEST_SAGA_ID }),
      ]);
    });

    it('leaves sagaId null and skips saga node when sagaId is omitted', async () => {
      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockSagaRepo.createIfNotExists).not.toHaveBeenCalled();
      expect(mockSagaRepo.save).not.toHaveBeenCalled();
      expect(savedEpisodes()).toEqual([expect.objectContaining({ sagaId: null })]);
    });

    it('groups saga nodes and tags each saga-bearing episode in the batch', async () => {
      const ep1 = makeEpisode('ep1');
      ep1.sagaId = KG_TEST_SAGA_ID;
      const ep2 = makeEpisode('ep2'); // no saga
      const ep3 = makeEpisode('ep3');
      ep3.sagaId = KG_TEST_SAGA_ID;

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [ep1, ep2, ep3],
      });

      // Saga createIfNotExists is called once per unique saga (grouped);
      // each saga-bearing episode carries the sagaId, the other stays null.
      expect(mockSagaRepo.createIfNotExists).toHaveBeenCalledTimes(1);
      const sagaIds = savedEpisodes().map((n) => n.sagaId);
      expect(sagaIds.filter((id) => id === KG_TEST_SAGA_ID)).toHaveLength(2);
      expect(sagaIds.filter((id) => id === null)).toHaveLength(1);
    });
  });

  // ─── Community update enqueue ─────────────────────────────────────────────

  describe('addEpisodes - community maintenance', () => {
    it('does not schedule maintenance by default', async () => {
      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockCommunityMaintenance.scheduleMaintenance).not.toHaveBeenCalled();
    });

    it('does not schedule when updateCommunities is true but no entities resolved', async () => {
      // Default resolveNodes mock returns empty resolvedNodes - no entities to
      // update. The call site skips the call.
      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
        updateCommunities: true,
      });

      expect(mockCommunityMaintenance.scheduleMaintenance).not.toHaveBeenCalled();
    });

    it('schedules maintenance after persist with the canonical entity ids', async () => {
      const resolved = KgNodeFactory.createEntityNode({ name: 'Alice' });
      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([resolved]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolved, nameEmbedding: [1, 0, 0] },
      ]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        resolvedNodes: [resolved],
        idMap: new Map(),
        duplicatePairs: [],
        candidates: [],
      });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
        updateCommunities: true,
      });

      const persistOrder = mockEntityNodeRepo.saveBulk.mock.invocationCallOrder[0];
      const scheduleOrder =
        mockCommunityMaintenance.scheduleMaintenance.mock.invocationCallOrder[0];
      expect(persistOrder).toBeLessThan(scheduleOrder);
      expect(mockCommunityMaintenance.scheduleMaintenance).toHaveBeenCalledWith(
        KG_TEST_USER_ID,
        KG_TEST_GRAPH_ID,
        [resolved.id],
      );
    });

    it('schedules once per distinct graphId, skipping graphs with no resolved entities', async () => {
      const otherGraphId = UuidSchema.parse('00000000-0000-4000-8000-000000000002');
      const resolvedA = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const resolvedB = {
        ...KgNodeFactory.createEntityNode({ name: 'Bob' }),
        graphId: otherGraphId,
      };
      const ep1 = makeEpisode('ep1');
      const ep2 = { ...makeEpisode('ep2'), graphId: otherGraphId };

      mockNodeExtraction.extractNodes
        .mockResolvedValueOnce(nodesResult([resolvedA]))
        .mockResolvedValueOnce(nodesResult([resolvedB]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolvedA, nameEmbedding: [1, 0, 0] },
        { ...resolvedB, nameEmbedding: [0, 1, 0] },
      ]);
      mockNodeResolution.resolveNodes
        .mockResolvedValueOnce({
          resolvedNodes: [resolvedA],
          idMap: new Map(),
          duplicatePairs: [],
          candidates: [],
        })
        .mockResolvedValueOnce({
          resolvedNodes: [resolvedB],
          idMap: new Map(),
          duplicatePairs: [],
          candidates: [],
        });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [ep1, ep2],
        updateCommunities: true,
      });

      expect(mockCommunityMaintenance.scheduleMaintenance).toHaveBeenCalledTimes(2);
      expect(mockCommunityMaintenance.scheduleMaintenance).toHaveBeenCalledWith(
        KG_TEST_USER_ID,
        KG_TEST_GRAPH_ID,
        [resolvedA.id],
      );
      expect(mockCommunityMaintenance.scheduleMaintenance).toHaveBeenCalledWith(
        KG_TEST_USER_ID,
        otherGraphId,
        [resolvedB.id],
      );
    });
  });
});
