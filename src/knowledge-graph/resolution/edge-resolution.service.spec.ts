import { createMock, DeepMocked } from '@golevelup/ts-jest';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Test, TestingModule } from '@nestjs/testing';

import { Uuid } from '@/common';
import { LLM_TRACER, NoOpLlmTracer } from '@/observability';
import {
  KG_HIGH_SIM_EMBEDDING,
  KG_NEAR_SAME_EMBEDDING,
  KG_REFERENCE_TIME,
  KG_TEST_GRAPH_ID,
  KgEdgeFactory,
  KgNodeFactory,
  u,
} from '@/test/factories';

import { EntityEdge } from '../models';
import { EntityEdgeRepository } from '../repository/repositories';
import { EdgeResolutionService } from './edge-resolution.service';
import { EdgeChunkSources } from './types';

// Stable test IDs so intra-batch dedup and endpoint matching reliably fire
// across edges constructed by `makeEdge` without explicit overrides.
const DEFAULT_SRC = u('src-id');
const DEFAULT_TGT = u('tgt-id');

const baseEpisode = KgNodeFactory.createEpisodicNode({
  name: 'Test Episode',
  content: 'Alice joined Acme Corp as CEO.',
  graphId: KG_TEST_GRAPH_ID,
});

function makeEdge(
  overrides: { name: string; fact: string } & Omit<Partial<EntityEdge>, 'name'>,
): EntityEdge {
  return KgEdgeFactory.createEntityEdge({
    sourceNodeId: DEFAULT_SRC,
    targetNodeId: DEFAULT_TGT,
    ...overrides,
  });
}

// Edges always carry origin-episode-qualified chunk sources through extraction +
// dedup; the dedup LLM path throws without them (single episode 0, chunk 0 here).
const chunkSources = (...edges: EntityEdge[]): EdgeChunkSources =>
  new Map(edges.map((e) => [e.id, { episodeIndex: 0, indices: new Set([0]) }]));

describe('EdgeResolutionService', () => {
  let service: EdgeResolutionService;
  let mockModel: DeepMocked<BaseChatModel>;
  let mockRunnable: { invoke: jest.Mock };
  let mockEdgeRepo: DeepMocked<EntityEdgeRepository>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        EdgeResolutionService,
        { provide: LLM_TRACER, useValue: new NoOpLlmTracer() },
      ],
    })
      .useMocker(createMock)
      .compile();

    service = module.get(EdgeResolutionService);
    mockEdgeRepo = module.get(EntityEdgeRepository);

    mockEdgeRepo.searchByFact.mockResolvedValue([]);
    jest.spyOn(service, 'collectCandidates').mockResolvedValue([]);

    mockModel = createMock<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  describe('dedupeEdges', () => {
    it('should collapse intra-batch exact duplicate to 1 survivor', async () => {
      const edge1 = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        episodes: [u('ep-1')],
      });
      const edge2 = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme', // same fact
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        episodes: [u('ep-2')],
      });

      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge1, edge2),
        [edge1, edge2],
        new Map(),
        KG_REFERENCE_TIME,
      );

      expect(result.survivors).toHaveLength(1);
      expect(result.survivors[0].episodes).toContain(u('ep-1'));
      expect(result.survivors[0].episodes).toContain(u('ep-2'));
    });

    it('should remap source/target ids via idMap', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        sourceNodeId: u('old-src-id'),
        targetNodeId: u('old-tgt-id'),
      });

      const idMap = new Map<Uuid, Uuid>([
        [u('old-src-id'), u('new-src-id')],
        [u('old-tgt-id'), u('new-tgt-id')],
      ]);

      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        idMap,
        KG_REFERENCE_TIME,
      );

      expect(result.survivors[0].sourceNodeId).toBe(u('new-src-id'));
      expect(result.survivors[0].targetNodeId).toBe(u('new-tgt-id'));
    });

    it('should add edge to survivors when no candidates exist', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
      });

      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
      expect(result.survivors).toHaveLength(1);
      expect(result.matchedExistingEdges).toHaveLength(0);
    });

    it('should route a duplicate to matchedExistingEdges with the episode appended', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        episodes: [baseEpisode.id],
      });
      const existingEdge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme Corp',
        factEmbedding: KG_NEAR_SAME_EMBEDDING,
      });
      existingEdge.id = u('exist-edge-id');

      // idx 0 is in endpoint range (1 endpoint edge)
      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [0],
        contradictedFacts: [],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue([existingEdge]);
      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      // The existing edge carries the new episode id so it can be re-persisted;
      // the extracted edge does not survive.
      expect(result.survivors).toHaveLength(0);
      expect(result.matchedExistingEdges).toHaveLength(1);
      expect(result.matchedExistingEdges[0].id).toBe(u('exist-edge-id'));
      expect(result.matchedExistingEdges[0].episodes).toContain(baseEpisode.id);
    });

    it('should record contradicted candidates per survivor', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is now CEO at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
      });
      const existingEdge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice was an engineer at Acme',
        factEmbedding: KG_NEAR_SAME_EMBEDDING,
      });
      existingEdge.id = u('old-edge-id');

      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [],
        contradictedFacts: [0],
      });

      jest.spyOn(service, 'collectCandidates').mockResolvedValue([existingEdge]);
      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      expect(result.survivors).toHaveLength(1);
      const contradictions = result.contradictionsBySurvivorId.get(
        result.survivors[0].id,
      );
      expect(contradictions).toHaveLength(1);
      expect(contradictions?.[0].id).toBe(u('old-edge-id'));
    });

    it('should reject duplicateFacts pointing into the similar range via the validator', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
      });
      const endpointEdge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is employed at Acme Corp',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        sourceNodeId: u('src-id'),
        targetNodeId: u('tgt-id'),
      });
      endpointEdge.id = u('endpoint-id');
      const similarEdge = makeEdge({
        name: 'EMPLOYED_AT',
        fact: 'Alice has a job at Acme',
        factEmbedding: KG_NEAR_SAME_EMBEDDING,
        sourceNodeId: u('other-src'),
        targetNodeId: u('other-tgt'),
      });
      similarEdge.id = u('similar-id');

      // idx 0 = endpoint edge, idx 1 = similar edge
      // duplicateFacts must only contain EXISTING FACTS range idx; the validator
      // surfaces this violation rather than silently filtering.
      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [1],
        contradictedFacts: [],
      });

      jest
        .spyOn(service, 'collectCandidates')
        .mockResolvedValue([endpointEdge, similarEdge]);
      await expect(
        service.dedupeEdges(
          mockModel,
          [baseEpisode],
          [[baseEpisode.content]],
          chunkSources(edge),
          [edge],
          new Map(),
          KG_REFERENCE_TIME,
        ),
      ).rejects.toThrow();
    });

    it('should set factEmbedding on surviving edges', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
      });

      const result = await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      expect(result.survivors[0].factEmbedding).toEqual(KG_HIGH_SIM_EMBEDDING);
    });

    it('should include keyword-only edge in similar candidates when no factEmbedding', async () => {
      const edge = makeEdge({ name: 'WORKS_AT', fact: 'Alice works at Acme' });
      // edge has no factEmbedding → cosine path skipped, but keyword path should find this
      const keywordEdge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is employed at Acme',
        factEmbedding: null,
        sourceNodeId: u('other-src'),
        targetNodeId: u('other-tgt'),
      });
      keywordEdge.id = u('keyword-id');

      mockEdgeRepo.searchByFact.mockResolvedValue([keywordEdge]);
      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [],
        contradictedFacts: [],
      });

      await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      // LLM should have been called with the keyword edge as a candidate
      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
    });

    it('should not include keyword result that is already an endpoint edge', async () => {
      const edge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
      });
      const endpointEdge = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is at Acme',
        factEmbedding: null,
      });
      endpointEdge.id = u('endpoint-id');

      // keyword search returns the endpoint edge - should be excluded from similarEdges
      mockEdgeRepo.searchByFact.mockResolvedValue([endpointEdge]);
      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [],
        contradictedFacts: [],
      });

      // candidate pool contains the endpoint edge (same src/tgt as `edge`)
      jest.spyOn(service, 'collectCandidates').mockResolvedValue([endpointEdge]);
      await service.dedupeEdges(
        mockModel,
        [baseEpisode],
        [[baseEpisode.content]],
        chunkSources(edge),
        [edge],
        new Map(),
        KG_REFERENCE_TIME,
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
    });
  });

  describe('invalidateEdges', () => {
    // Pure arithmetic over the enriched survivors + the contradictions recorded
    // by dedupeEdges. No model call.
    it('does not invalidate when both edges lack validAt', () => {
      const survivor = makeEdge({ name: 'WORKS_AT', fact: 'Alice is now CEO at Acme' });
      const existing = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice was an engineer at Acme',
      });
      existing.id = u('old-edge-id');

      const { invalidatedEdges } = service.invalidateEdges(
        [survivor],
        new Map([[survivor.id, [existing]]]),
      );

      expect(invalidatedEdges).toHaveLength(0);
    });

    it('invalidates a predating contradicted edge at the survivor validAt', () => {
      const survivor = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is now CEO at Acme',
        validAt: new Date('2024-06-01'),
      });
      const existing = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice was an engineer at Acme',
        validAt: new Date('2023-01-01'),
      });
      existing.id = u('old-edge-id');

      const { invalidatedEdges, invalidatedBySurvivorId } = service.invalidateEdges(
        [survivor],
        new Map([[survivor.id, [existing]]]),
      );

      expect(invalidatedEdges).toHaveLength(1);
      expect(invalidatedEdges[0].id).toBe(u('old-edge-id'));
      expect(invalidatedEdges[0].invalidAt).toEqual(new Date('2024-06-01'));
      expect(invalidatedEdges[0].expiredAt).toBeInstanceOf(Date);
      expect(invalidatedBySurvivorId.get(survivor.id)).toHaveLength(1);
    });

    it('expires a survivor that already carries an invalidAt (guard a)', () => {
      const survivor = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice left Acme',
        validAt: new Date('2024-01-01'),
        invalidAt: new Date('2024-06-01'),
      });
      expect(survivor.expiredAt).toBeNull();

      service.invalidateEdges([survivor], new Map());

      expect(survivor.expiredAt).toBeInstanceOf(Date);
    });

    it('self-expires a survivor superseded by a postdating contradiction (guard b)', () => {
      const survivor = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is an engineer at Acme',
        validAt: new Date('2023-01-01'),
      });
      const later = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice is CEO at Acme',
        validAt: new Date('2024-06-01'),
      });
      later.id = u('later-edge-id');

      const { invalidatedEdges } = service.invalidateEdges(
        [survivor],
        new Map([[survivor.id, [later]]]),
      );

      expect(survivor.invalidAt).toEqual(new Date('2024-06-01'));
      expect(survivor.expiredAt).toBeInstanceOf(Date);
      // `later` postdates the survivor, so it is not itself invalidated.
      expect(invalidatedEdges).toHaveLength(0);
    });
  });

  describe('dedupeAcrossBatch', () => {
    it('collapses a cross-episode duplicate into one flat canonical edge', async () => {
      // Same endpoints + fact extracted from two different batch episodes.
      const edge1 = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        episodes: [u('ep-1')],
      });
      const edge2 = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: KG_HIGH_SIM_EMBEDDING,
        episodes: [u('ep-2')],
      });

      // Each edge sees the other as a same-endpoint candidate (idx 0); the LLM
      // flags it as a duplicate.
      mockRunnable.invoke.mockResolvedValue({
        duplicateFacts: [0],
        contradictedFacts: [],
      });

      // edge1 originates in episode 0, edge2 in episode 1.
      const sources: EdgeChunkSources = new Map([
        [edge1.id, { episodeIndex: 0, indices: new Set([0]) }],
        [edge2.id, { episodeIndex: 1, indices: new Set([0]) }],
      ]);

      const result = await service.dedupeAcrossBatch(
        mockModel,
        [[edge1], [edge2]],
        [baseEpisode, baseEpisode],
        [['chunk 0'], ['chunk 1']],
        sources,
        [[], []],
      );

      // Flat list, one surviving canonical edge whose episodes union both origins.
      expect(result).toHaveLength(1);
      expect([edge1.id, edge2.id]).toContain(result[0].id);
      expect(result[0].episodes).toEqual(expect.arrayContaining([u('ep-1'), u('ep-2')]));
    });

    it('returns both edges flat when they are not duplicates', async () => {
      // Different endpoints + no embeddings -> no endpoint/cosine candidates, so
      // no LLM dedup call is even made.
      const edge1 = makeEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme',
        factEmbedding: null,
        sourceNodeId: u('a'),
        targetNodeId: u('b'),
        episodes: [u('ep-1')],
      });
      const edge2 = makeEdge({
        name: 'LIVES_IN',
        fact: 'Bob lives in Paris',
        factEmbedding: null,
        sourceNodeId: u('c'),
        targetNodeId: u('d'),
        episodes: [u('ep-2')],
      });

      const sources: EdgeChunkSources = new Map([
        [edge1.id, { episodeIndex: 0, indices: new Set([0]) }],
        [edge2.id, { episodeIndex: 1, indices: new Set([0]) }],
      ]);

      const result = await service.dedupeAcrossBatch(
        mockModel,
        [[edge1], [edge2]],
        [baseEpisode, baseEpisode],
        [['chunk 0'], ['chunk 1']],
        sources,
        [[], []],
      );

      expect(result.map((e) => e.id).sort()).toEqual([edge1.id, edge2.id].sort());
      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
    });
  });
});
