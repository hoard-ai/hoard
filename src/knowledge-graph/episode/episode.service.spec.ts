import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import type { Uuid } from '@/common/schemas';
import { UuidSchema } from '@/common/schemas';
import { KnowledgeGraphConfigService } from '@/config/knowledge-graph';
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
import type { EntityEdge, EntityNode } from '../models';
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
  chunkIndicesByExtractedId: new Map<Uuid, Set<number>>(
    nodes.map((n) => [n.id, new Set([0])]),
  ),
  corefByExtractedId: new Map(),
  unresolvedReferences: [],
});
const edgesResult = (edges: EntityEdge[]) => ({
  edges,
  chunkIndicesByEdgeId: new Map<Uuid, Set<number>>(
    edges.map((e) => [e.id, new Set([0])]),
  ),
  committedCorefBindings: [],
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
      providers: [
        EpisodeService,
        { provide: LLM_TRACER, useValue: new NoOpLlmTracer() },
        {
          provide: KnowledgeGraphConfigService,
          useValue: { memoryBackpressureConcurrencyLimit: 10 },
        },
      ],
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
      newNodes: [],
      nodesMatchedToPreexistingNodes: [],
      preexistingCandidates: [],
    });
    mockNodeResolution.dedupeAcrossBatch.mockReturnValue([]);
    mockNodeResolution.canonicalizeEpisodeNodes.mockResolvedValue([]);
    mockEmbeddingService.embedNodes.mockResolvedValue([]);
    mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([]));
    mockEdgeExtraction.enrichEdges.mockResolvedValue(undefined);
    mockEmbeddingService.embedEdges.mockResolvedValue([]);
    mockEdgeResolution.dedupeEdges.mockResolvedValue({
      matchedPreexistingEdges: [],
      newEdges: [],
      contradictionsByNewEdgeId: new Map(),
    });
    mockEdgeResolution.invalidateEdges.mockReturnValue({
      invalidatedEdges: [],
      invalidatedByNewEdgeId: new Map(),
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

    it('extracts edges with canonical nodes (resolved + matched preexisting)', async () => {
      const resolved = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const preexisting = {
        ...KgNodeFactory.createEntityNode({ name: 'Bob' }),
        id: u('preexisting-bob-id'),
      };
      const alias = KgNodeFactory.createEntityNode({ name: 'Robert' });

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([resolved, alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolved, nameEmbedding: null },
        { ...alias, nameEmbedding: null },
      ]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [resolved],
        nodesMatchedToPreexistingNodes: [
          { extractedId: alias.id, preexistingNodeId: preexisting.id },
        ],
        preexistingCandidates: [preexisting],
      });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockEdgeExtraction.extractEdges).toHaveBeenCalledWith(
        mockModel,
        expect.anything(),
        expect.any(Array),
        expect.arrayContaining([resolved, preexisting]),
        [],
        undefined,
        undefined,
        undefined,
        false, // resolveCoreferences
        expect.any(Array), // corefCandidates
        expect.any(Array), // unresolvedReferences
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

    it('calls dedupeEdges with embedded edges and canonicalIdByNodeId', async () => {
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

      expect(mockEdgeResolution.dedupeEdges).toHaveBeenCalledWith(
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

    it('pools newEdges from dedupe into enrichEdges, then invalidates them', async () => {
      const edge = KgEdgeFactory.createEntityEdge({
        name: 'WORKS_AT',
        sourceNodeId: u('src'),
        targetNodeId: u('tgt'),
        fact: 'Alice works at Acme Corp',
      });
      const embeddedEdge = { ...edge, factEmbedding: [1, 0, 0] };

      mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([edge]));
      mockEmbeddingService.embedEdges.mockResolvedValue([embeddedEdge]);
      mockEdgeResolution.dedupeEdges.mockResolvedValue({
        matchedPreexistingEdges: [],
        newEdges: [embeddedEdge],
        contradictionsByNewEdgeId: new Map(),
      });

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockEdgeExtraction.enrichEdges).toHaveBeenCalledWith(
        mockModel,
        [embeddedEdge],
        expect.any(Array), // canonicalNodes
        expect.any(Array), // episodes
        expect.any(Array), // chunksPerEpisode
        expect.any(Map), // chunkSources
        undefined,
        undefined,
        expect.any(Array), // committedCorefBindingsPerEpisode
        expect.anything(),
      );
      expect(mockEdgeResolution.invalidateEdges).toHaveBeenCalledWith(
        [embeddedEdge],
        expect.any(Map),
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
        expect(entry.unresolvedReferences).toBeInstanceOf(Array);
      });
    });

    it('drops only the claimed reference when two share a surface form', async () => {
      // Two distinct "she" occurrences; an edge extraction resolves only one.
      const sheKept = {
        id: u('she-kept'),
        surfaceForm: 'she',
        locatingQuote: 'she made the first dive',
        sourceChunkIndex: 0,
      };
      const sheClaimed = {
        id: u('she-claimed'),
        surfaceForm: 'she',
        locatingQuote: 'she named the largest colony',
        sourceChunkIndex: 0,
      };
      mockNodeExtraction.extractNodes.mockResolvedValue({
        ...nodesResult([]),
        unresolvedReferences: [sheKept, sheClaimed],
      });
      mockEdgeExtraction.extractEdges.mockResolvedValue({
        ...edgesResult([]),
        committedCorefBindings: [
          {
            surfaceForm: 'she',
            boundNodeId: u('marquez'),
            sourceChunkIndex: 0,
            locatingQuote: 'she named the largest',
            resolvedUnresolvedReferenceId: sheClaimed.id,
          },
        ],
      });

      const [result] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      // A surface-form join would clobber BOTH "she"s; the id join drops only the
      // claimed occurrence and strips the survivor's internal id/chunk.
      expect(result.unresolvedReferences).toEqual([
        { surfaceForm: 'she', locatingQuote: 'she made the first dive' },
      ]);
    });

    it('builds one episodic edge per canonical node referenced by each episode', async () => {
      const resolved = KgNodeFactory.createEntityNode({ name: 'Alice' });
      const alias = KgNodeFactory.createEntityNode({ name: 'Bobby' });
      const preexisting = {
        ...KgNodeFactory.createEntityNode({ name: 'Bob' }),
        id: u('bob-id'),
      };

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([resolved, alias]));
      mockEmbeddingService.embedNodes.mockResolvedValue([
        { ...resolved, nameEmbedding: null },
        { ...alias, nameEmbedding: null },
      ]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [resolved],
        nodesMatchedToPreexistingNodes: [
          { extractedId: alias.id, preexistingNodeId: preexisting.id },
        ],
        preexistingCandidates: [preexisting],
      });

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(entry.episodicEdges).toHaveLength(2);
      expect(entry.episodicEdges.map((e) => e.targetNodeId)).toEqual(
        expect.arrayContaining([resolved.id, preexisting.id]),
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
          newNodes: [canonical],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
        })
        .mockResolvedValueOnce({
          newNodes: [],
          nodesMatchedToPreexistingNodes: [
            { extractedId: alias.id, preexistingNodeId: canonical.id },
          ],
          preexistingCandidates: [],
        });

      const result = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1'), makeEpisode('ep2')],
      });

      const allNodes = result.flatMap((r) => r.nodes);
      expect(allNodes.find((n) => n.id === alias.id)).toBeUndefined();
      expect(allNodes.find((n) => n.id === canonical.id)).toBeDefined();
    });

    it('preexisting node referenced as canonical target is pulled into the matching episode entry', async () => {
      const preexistingCanonical = KgNodeFactory.createEntityNode({
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
        newNodes: [],
        nodesMatchedToPreexistingNodes: [
          { extractedId: alias.id, preexistingNodeId: preexistingCanonical.id },
        ],
        preexistingCandidates: [preexistingCanonical],
      });

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(entry.nodes.find((n) => n.id === preexistingCanonical.id)).toBeDefined();
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
          newNodes: [canonical],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
        })
        .mockResolvedValueOnce({
          newNodes: [canonical],
          nodesMatchedToPreexistingNodes: [
            { extractedId: alias.id, preexistingNodeId: canonical.id },
          ],
          preexistingCandidates: [],
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
  // node-resolution.service.spec.ts ('dedupeAcrossBatch'); the canonical
  // projection helpers (recompute*) are unit-tested in episode-utils.spec.ts.
  // The orchestration test below verifies that pairs returned by the service
  // participate in the final canonical projection.

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
          newNodes: [canonical],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
        })
        .mockResolvedValueOnce({
          newNodes: [alias],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
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

  // ─── Within-episode canonicalization (phantom-node fold) ───────────────────

  describe('addEpisodes - within-episode canonicalization', () => {
    // Long enough to clear CHUNK_MAX_TOKENS so prepareChunks splits it - the
    // canonicalization pass only runs for multi-chunk episodes.
    const longEpisode = (name: string) => ({
      ...makeEpisode(name),
      content: 'This is a filler sentence about nothing in particular. '.repeat(200),
    });

    it('skips the canonicalization call for single-chunk episodes', async () => {
      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [makeEpisode('ep1')],
      });

      expect(mockNodeResolution.canonicalizeEpisodeNodes).not.toHaveBeenCalled();
    });

    it('folds returned pairs: the merged-away node vanishes and its edges are remapped', async () => {
      const named = KgNodeFactory.createEntityNode({ name: 'Dr. Elena Marquez' });
      const phantom = KgNodeFactory.createEntityNode({ name: 'biologist' });
      const place = KgNodeFactory.createEntityNode({ name: 'Valparaiso' });

      mockNodeExtraction.extractNodes.mockResolvedValue(
        nodesResult([named, phantom, place]),
      );
      mockEmbeddingService.embedNodes.mockResolvedValue([named, phantom, place]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [named, phantom, place],
        nodesMatchedToPreexistingNodes: [],
        preexistingCandidates: [],
      });
      mockNodeResolution.canonicalizeEpisodeNodes.mockResolvedValue([
        [phantom.id, named.id],
      ]);
      const edge = KgEdgeFactory.createEntityEdge({
        name: 'WORKS_AT',
        sourceNodeId: phantom.id,
        targetNodeId: place.id,
        fact: 'The biologist works in Valparaiso',
      });
      mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([edge]));
      mockEmbeddingService.embedEdges.mockImplementation((edges) =>
        Promise.resolve(edges),
      );

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [longEpisode('ep1')],
      });

      expect(mockNodeResolution.canonicalizeEpisodeNodes).toHaveBeenCalledTimes(1);
      expect(entry.nodes.map((n) => n.id)).toEqual(
        expect.arrayContaining([named.id, place.id]),
      );
      expect(entry.nodes.find((n) => n.id === phantom.id)).toBeUndefined();

      const [embedded] = mockEmbeddingService.embedEdges.mock.calls[0];
      expect(embedded).toHaveLength(1);
      expect(embedded[0].sourceNodeId).toBe(named.id);
    });

    it('pulls post-remap self-loops out of the edge pipeline into summary fact context', async () => {
      const named = KgNodeFactory.createEntityNode({ name: 'Dr. Elena Marquez' });
      const phantom = KgNodeFactory.createEntityNode({ name: 'biologist' });

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([named, phantom]));
      mockEmbeddingService.embedNodes.mockResolvedValue([named, phantom]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [named, phantom],
        nodesMatchedToPreexistingNodes: [],
        preexistingCandidates: [],
      });
      mockNodeResolution.canonicalizeEpisodeNodes.mockResolvedValue([
        [phantom.id, named.id],
      ]);
      const selfLoop = KgEdgeFactory.createEntityEdge({
        name: 'HAS_ROLE',
        sourceNodeId: named.id,
        targetNodeId: phantom.id,
        fact: 'Elena is the expedition biologist',
      });
      mockEdgeExtraction.extractEdges.mockResolvedValue(edgesResult([selfLoop]));
      mockEmbeddingService.embedEdges.mockImplementation((edges) =>
        Promise.resolve(edges),
      );

      await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [longEpisode('ep1')],
      });

      // The collapsed edge never reaches embedding/dedup/persist...
      expect(mockEmbeddingService.embedEdges).toHaveBeenCalledWith([]);
      const savedEdges = mockEntityEdgeRepo.saveBulk.mock.calls.flatMap(
        ([edges]) => edges,
      );
      expect(savedEdges.find((e) => e.id === selfLoop.id)).toBeUndefined();

      // ...but its fact feeds the node summary context.
      const summaryFacts = mockNodeExtraction.summarizeNodes.mock.calls[0][2];
      expect(summaryFacts).toEqual([
        expect.objectContaining({ fact: 'Elena is the expedition biologist' }),
      ]);
    });

    it('flips a pair whose merged-away node is live: the live id survives under the new name', async () => {
      const newNode = KgNodeFactory.createEntityNode({ name: 'Dr. Elena Marquez' });
      const phantomExtracted = KgNodeFactory.createEntityNode({ name: 'biologist' });
      const liveNode = KgNodeFactory.createEntityNode({ name: 'biologist' });

      mockNodeExtraction.extractNodes.mockResolvedValue(
        nodesResult([newNode, phantomExtracted]),
      );
      mockEmbeddingService.embedNodes.mockResolvedValue([newNode, phantomExtracted]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [newNode],
        nodesMatchedToPreexistingNodes: [
          { extractedId: phantomExtracted.id, preexistingNodeId: liveNode.id },
        ],
        preexistingCandidates: [liveNode],
      });
      mockNodeResolution.canonicalizeEpisodeNodes.mockResolvedValue([
        [liveNode.id, newNode.id],
      ]);

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [longEpisode('ep1')],
      });

      expect(entry.nodes).toHaveLength(1);
      expect(entry.nodes[0]).toEqual(
        expect.objectContaining({ id: liveNode.id, name: 'Dr. Elena Marquez' }),
      );
      expect(entry.nodes.find((n) => n.id === newNode.id)).toBeUndefined();
    });

    it('drops a pair between two live nodes and keeps both', async () => {
      const p1 = KgNodeFactory.createEntityNode({ name: 'the captain' });
      const p2 = KgNodeFactory.createEntityNode({ name: 'Captain Ruiz' });
      const live1 = KgNodeFactory.createEntityNode({ name: 'the captain' });
      const live2 = KgNodeFactory.createEntityNode({ name: 'Captain Ruiz' });

      mockNodeExtraction.extractNodes.mockResolvedValue(nodesResult([p1, p2]));
      mockEmbeddingService.embedNodes.mockResolvedValue([p1, p2]);
      mockNodeResolution.resolveNodes.mockResolvedValue({
        newNodes: [],
        nodesMatchedToPreexistingNodes: [
          { extractedId: p1.id, preexistingNodeId: live1.id },
          { extractedId: p2.id, preexistingNodeId: live2.id },
        ],
        preexistingCandidates: [live1, live2],
      });
      mockNodeResolution.canonicalizeEpisodeNodes.mockResolvedValue([
        [live1.id, live2.id],
      ]);

      const [entry] = await service.addTextEpisodes({
        userId: KG_TEST_USER_ID,
        episodes: [longEpisode('ep1')],
      });

      expect(entry.nodes.map((n) => n.id).sort()).toEqual([live1.id, live2.id].sort());
      expect(entry.nodes.map((n) => n.name).sort()).toEqual(
        ['Captain Ruiz', 'the captain'].sort(),
      );
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
      // Default resolveNodes mock returns empty newNodes - no entities to
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
        newNodes: [resolved],
        nodesMatchedToPreexistingNodes: [],
        preexistingCandidates: [],
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
          newNodes: [resolvedA],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
        })
        .mockResolvedValueOnce({
          newNodes: [resolvedB],
          nodesMatchedToPreexistingNodes: [],
          preexistingCandidates: [],
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

  // ─── Episode deletion ─────────────────────────────────────────────────────

  describe('deleteEpisode', () => {
    it('schedules community maintenance for the mentioned entities after deletion', async () => {
      const episode = KgNodeFactory.createEpisodicNode();
      const mentionedIds = [u('mention-1'), u('mention-2')];
      mockEpisodicNodeRepo.getById.mockResolvedValue(episode);
      mockEpisodicNodeRepo.getMentionedEntityIds.mockResolvedValue(mentionedIds);
      mockEntityEdgeRepo.getIdsForEpisodeDeletion.mockResolvedValue([]);

      await service.deleteEpisode(KG_TEST_USER_ID, episode.id);

      const deleteOrder = mockEpisodicNodeRepo.delete.mock.invocationCallOrder[0];
      const scheduleOrder =
        mockCommunityMaintenance.scheduleMaintenance.mock.invocationCallOrder[0];
      expect(deleteOrder).toBeLessThan(scheduleOrder);
      expect(mockCommunityMaintenance.scheduleMaintenance).toHaveBeenCalledWith(
        KG_TEST_USER_ID,
        episode.graphId,
        mentionedIds,
      );
    });

    it('does not schedule maintenance when the episode mentions no entities', async () => {
      const episode = KgNodeFactory.createEpisodicNode();
      mockEpisodicNodeRepo.getById.mockResolvedValue(episode);
      mockEpisodicNodeRepo.getMentionedEntityIds.mockResolvedValue([]);
      mockEntityEdgeRepo.getIdsForEpisodeDeletion.mockResolvedValue([]);

      await service.deleteEpisode(KG_TEST_USER_ID, episode.id);

      expect(mockCommunityMaintenance.scheduleMaintenance).not.toHaveBeenCalled();
    });

    it('does not schedule maintenance when the episode does not exist', async () => {
      mockEpisodicNodeRepo.getById.mockResolvedValue(null);

      await service.deleteEpisode(KG_TEST_USER_ID, u('missing-episode'));

      expect(mockCommunityMaintenance.scheduleMaintenance).not.toHaveBeenCalled();
    });
  });
});
