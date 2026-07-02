import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { mockDeep } from 'jest-mock-extended';
import { z } from 'zod';

import { NoOpLlmTracer } from '@/observability';
import { KG_TEST_GRAPH_ID, KgEdgeFactory, KgNodeFactory } from '@/test/factories';

import type { EdgeTypeMap, EdgeTypeMappings } from '../episode/types';
import { EntityEdge } from '../models';
import { EdgeChunkSources } from '../resolution/types';
import { NodeLabelSchema, RelationshipTypeSchema } from '../types';
import { EdgeExtractionService } from './edge-extraction.service';

const baseEpisode = KgNodeFactory.createEpisodicNode({
  name: 'Test Episode',
  content: 'Alice works at Acme Corp. Bob is the CEO of Acme Corp.',
  graphId: KG_TEST_GRAPH_ID,
});

const aliceNode = KgNodeFactory.createEntityNode({
  name: 'Alice',
  graphId: KG_TEST_GRAPH_ID,
});
const bobNode = KgNodeFactory.createEntityNode({
  name: 'Bob',
  graphId: KG_TEST_GRAPH_ID,
});
const acmeNode = KgNodeFactory.createEntityNode({
  name: 'Acme Corp',
  graphId: KG_TEST_GRAPH_ID,
});
const nodes = [aliceNode, bobNode, acmeNode];

describe('EdgeExtractionService', () => {
  let service: EdgeExtractionService;
  let mockModel: ReturnType<typeof mockDeep<BaseChatModel>>;
  let mockRunnable: { invoke: jest.Mock };

  beforeEach(() => {
    service = new EdgeExtractionService(new NoOpLlmTracer());
    mockModel = mockDeep<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  describe('extractEdges', () => {
    it('should return EntityEdge[] matching source/target ids', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 0,
            targetEntityIdx: 2,
            relationType: 'WORKS_AT',
            fact: 'Alice works at Acme Corp.',
          },
        ],
      });

      const { edges } = await service.extractEdges(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        nodes,
        [],
      );

      expect(edges).toHaveLength(1);
      expect(edges[0].sourceNodeId).toBe(aliceNode.id);
      expect(edges[0].targetNodeId).toBe(acmeNode.id);
      expect(edges[0].name).toBe('WORKS_AT');
      expect(edges[0].fact).toBe('Alice works at Acme Corp.');
      expect(edges[0].graphId).toBe(KG_TEST_GRAPH_ID);
    });

    it('extracts edges with null temporal bounds (timestamps are filled later by enrichEdges)', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 0,
            targetEntityIdx: 2,
            relationType: 'WORKS_AT',
            fact: 'Alice works at Acme Corp.',
          },
        ],
      });

      const { edges } = await service.extractEdges(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        nodes,
        [],
      );

      expect(edges[0].validAt).toBeNull();
      expect(edges[0].invalidAt).toBeNull();
    });

    it('tags each edge with the chunk index it was extracted from', async () => {
      mockRunnable.invoke
        .mockResolvedValueOnce({
          edges: [
            {
              sourceEntityIdx: 0,
              targetEntityIdx: 2,
              relationType: 'WORKS_AT',
              fact: 'Alice works at Acme Corp.',
            },
          ],
        })
        .mockResolvedValueOnce({
          edges: [
            {
              sourceEntityIdx: 1,
              targetEntityIdx: 2,
              relationType: 'CEO_OF',
              fact: 'Bob is the CEO of Acme Corp.',
            },
          ],
        });

      const { edges, chunkIndicesByEdgeId } = await service.extractEdges(
        mockModel,
        baseEpisode,
        ['Alice works at Acme Corp.', 'Bob is the CEO of Acme Corp.'],
        nodes,
        [],
      );

      expect(edges).toHaveLength(2);
      const worksAt = edges.find((e) => e.name === 'WORKS_AT')!;
      const ceoOf = edges.find((e) => e.name === 'CEO_OF')!;
      // Singleton per origin chunk - dedup never unions edge chunk sources.
      expect([...chunkIndicesByEdgeId.get(worksAt.id)!]).toEqual([0]);
      expect([...chunkIndicesByEdgeId.get(ceoOf.id)!]).toEqual([1]);
    });

    it('should reject edges with out-of-range source idx via the validator', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 99,
            targetEntityIdx: 2,
            relationType: 'WORKS_AT',
            fact: 'Someone works at Acme.',
          },
        ],
      });

      await expect(
        service.extractEdges(mockModel, baseEpisode, [baseEpisode.content], nodes, []),
      ).rejects.toThrow();
    });

    it('should reject edges with out-of-range target idx via the validator', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 0,
            targetEntityIdx: 99,
            relationType: 'WORKS_AT',
            fact: 'Alice works somewhere.',
          },
        ],
      });

      await expect(
        service.extractEdges(mockModel, baseEpisode, [baseEpisode.content], nodes, []),
      ).rejects.toThrow();
    });

    it('should set episodes to [episode.id] on each extracted edge', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 0,
            targetEntityIdx: 2,
            relationType: 'WORKS_AT',
            fact: 'Alice works at Acme Corp.',
          },
        ],
      });

      const { edges } = await service.extractEdges(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        nodes,
        [],
      );

      expect(edges[0].episodes).toEqual([baseEpisode.id]);
    });

    it('should return empty array when no edges extracted', async () => {
      mockRunnable.invoke.mockResolvedValue({ edges: [] });

      const { edges } = await service.extractEdges(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        nodes,
        [],
      );

      expect(edges).toEqual([]);
    });

    it('should assign id to each returned edge', async () => {
      mockRunnable.invoke.mockResolvedValue({
        edges: [
          {
            sourceEntityIdx: 0,
            targetEntityIdx: 2,
            relationType: 'WORKS_AT',
            fact: 'Alice works at Acme Corp.',
          },
          {
            sourceEntityIdx: 1,
            targetEntityIdx: 2,
            relationType: 'CEO_OF',
            fact: 'Bob is CEO of Acme Corp.',
          },
        ],
      });

      const { edges } = await service.extractEdges(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        nodes,
        [],
      );

      expect(edges).toHaveLength(2);
      edges.forEach((e) => expect(e.id).toBeTruthy());
      expect(edges[0].id).not.toBe(edges[1].id);
    });
  });

  describe('enrichEdges', () => {
    const makeSurvivor = (): EntityEdge =>
      KgEdgeFactory.createEntityEdge({
        name: 'WORKS_AT',
        fact: 'Alice works at Acme Corp.',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId: aliceNode.id,
        targetNodeId: acmeNode.id,
      });

    const sourcesFor = (edge: EntityEdge): EdgeChunkSources =>
      new Map([[edge.id, { episodeIndex: 0, indices: new Set([0]) }]]);

    it('fills temporal bounds on an untyped survivor', async () => {
      const edge = makeSurvivor();
      mockRunnable.invoke.mockResolvedValue({
        validAt: '2024-01-01T00:00:00.000Z',
        invalidAt: null,
      });

      await service.enrichEdges(
        mockModel,
        [edge],
        nodes,
        [baseEpisode],
        [[baseEpisode.content]],
        sourcesFor(edge),
      );

      expect(edge.validAt).toEqual(new Date('2024-01-01T00:00:00.000Z'));
      expect(edge.invalidAt).toBeNull();
    });

    it('fills temporal bounds and merges custom attributes for a typed survivor', async () => {
      const edge = makeSurvivor();
      // aliceNode/acmeNode default to the 'Entity' label, so WORKS_AT maps for
      // the [Entity, Entity] endpoint pair.
      const edgeTypes: EdgeTypeMap = {
        [RelationshipTypeSchema.parse('WORKS_AT')]: {
          description: 'employment',
          schema: z.object({ role: z.string() }),
        },
      };
      const edgeTypeMappings: EdgeTypeMappings = new Map([
        [
          [NodeLabelSchema.parse('Entity'), NodeLabelSchema.parse('Entity')],
          [RelationshipTypeSchema.parse('WORKS_AT')],
        ],
      ]);
      mockRunnable.invoke.mockResolvedValue({
        validAt: '2024-01-01T00:00:00.000Z',
        invalidAt: null,
        attributes: { role: 'CEO' },
      });

      await service.enrichEdges(
        mockModel,
        [edge],
        nodes,
        [baseEpisode],
        [[baseEpisode.content]],
        sourcesFor(edge),
        edgeTypes,
        edgeTypeMappings,
      );

      expect(edge.validAt).toEqual(new Date('2024-01-01T00:00:00.000Z'));
      expect(edge.attributes).toMatchObject({ role: 'CEO' });
    });

    it('throws when a survivor has no chunk source', async () => {
      const edge = makeSurvivor();
      await expect(
        service.enrichEdges(
          mockModel,
          [edge],
          nodes,
          [baseEpisode],
          [[baseEpisode.content]],
          new Map(),
        ),
      ).rejects.toThrow();
    });
  });
});
