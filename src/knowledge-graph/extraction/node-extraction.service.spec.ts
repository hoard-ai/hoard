import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { mockDeep } from 'jest-mock-extended';
import { z } from 'zod';

import { Uuid } from '@/common/schemas';
import { NoOpLlmTracer } from '@/observability';
import { KG_TEST_GRAPH_ID, KgEdgeFactory, KgNodeFactory, u } from '@/test/factories';

import { EpisodicNode } from '../models';
import { NodeExtractionService } from './node-extraction.service';

const baseEpisode = KgNodeFactory.createEpisodicNode({
  name: 'Test Episode',
  content: 'Alice works at Acme Corp and knows Bob.',
  graphId: KG_TEST_GRAPH_ID,
});

describe('NodeExtractionService', () => {
  let service: NodeExtractionService;
  let mockModel: ReturnType<typeof mockDeep<BaseChatModel>>;
  let mockRunnable: { invoke: jest.Mock };

  beforeEach(() => {
    service = new NodeExtractionService(new NoOpLlmTracer());
    mockModel = mockDeep<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  it('should return EntityNode[] with correct names and graphId', async () => {
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice' }, { name: 'Acme Corp' }, { name: 'Bob' }],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
    );

    expect(nodes).toHaveLength(3);
    expect(nodes.map((n) => n.name)).toEqual(['Alice', 'Acme Corp', 'Bob']);
    nodes.forEach((n) => expect(n.graphId).toBe(KG_TEST_GRAPH_ID));
  });

  it('should filter whitespace-only names (schema rejects empty strings upstream)', async () => {
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice' }, { name: '   ' }, { name: 'Bob' }],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
    );

    expect(nodes).toHaveLength(2);
    expect(nodes.map((n) => n.name)).toEqual(['Alice', 'Bob']);
  });

  it('should assign default Entity label when no entityTypes', async () => {
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice' }],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
    );

    expect(nodes[0].labels).toEqual(['Entity']);
  });

  it('should assign correct labels from entityTypes map', async () => {
    const entityTypes = {
      Person: { description: 'A human individual', schema: z.object({}) },
      Organization: { description: 'A company or group', schema: z.object({}) },
    };
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [
        { name: 'Alice', entityTypeId: 0 },
        { name: 'Acme Corp', entityTypeId: 1 },
      ],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
      entityTypes,
    );

    expect(nodes[0].labels).toEqual(['Entity', 'Person']);
    expect(nodes[1].labels).toEqual(['Entity', 'Organization']);
  });

  it('should reject out-of-range entityTypeId via the validator and surface after retries', async () => {
    const entityTypes = {
      Person: { description: 'A human individual', schema: z.object({}) },
    };
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice', entityTypeId: 99 }],
    });

    await expect(
      service.extractNodes(
        mockModel,
        baseEpisode,
        [baseEpisode.content],
        [],
        entityTypes,
      ),
    ).rejects.toThrow();
  });

  it('should assign id to each returned node', async () => {
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice' }, { name: 'Bob' }],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
    );

    nodes.forEach((n) => expect(n.id).toBeTruthy());
    expect(nodes[0].id).not.toBe(nodes[1].id);
  });

  it('should return empty array when no entities extracted', async () => {
    mockRunnable.invoke.mockResolvedValue({ extractedEntities: [] });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
    );

    expect(nodes).toEqual([]);
  });

  it('unions chunk indices for a node extracted from multiple chunks', async () => {
    // Alice appears in both chunks (deduped by name); Bob only in the first.
    mockRunnable.invoke
      .mockResolvedValueOnce({
        extractedEntities: [{ name: 'Alice' }, { name: 'Bob' }],
      })
      .mockResolvedValueOnce({ extractedEntities: [{ name: 'Alice' }] });

    const { nodes, chunkIndicesByNodeId } = await service.extractNodes(
      mockModel,
      baseEpisode,
      ['Alice met Bob.', 'Alice left.'],
      [],
    );

    expect(nodes.map((n) => n.name)).toEqual(['Alice', 'Bob']);
    const alice = nodes.find((n) => n.name === 'Alice')!;
    const bob = nodes.find((n) => n.name === 'Bob')!;
    expect([...chunkIndicesByNodeId.get(alice.id)!].sort()).toEqual([0, 1]);
    expect([...chunkIndicesByNodeId.get(bob.id)!]).toEqual([0]);
  });

  it('does not run attribute extraction (moved to post-resolution step in EpisodeService)', async () => {
    const entityTypes = {
      Person: {
        description: 'A human individual',
        schema: z.object({ age: z.number().optional() }),
      },
    };
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice', entityTypeId: 0 }],
    });

    const { nodes } = await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
      entityTypes,
    );

    // Only the entity-extraction LLM call; attribute extraction happens post-resolution
    expect(mockModel.withStructuredOutput).toHaveBeenCalledTimes(1);
    expect(nodes[0].attributes).toEqual({});
  });

  it('should not call attribute extraction LLM when entity type has schema but extraction is skipped', async () => {
    const entityTypes = {
      Person: { description: 'A human individual', schema: z.object({}) },
    };
    mockRunnable.invoke.mockResolvedValue({
      extractedEntities: [{ name: 'Alice', entityTypeId: 0 }],
    });

    await service.extractNodes(
      mockModel,
      baseEpisode,
      [baseEpisode.content],
      [],
      entityTypes,
    );

    expect(mockModel.withStructuredOutput).toHaveBeenCalledTimes(1);
  });

  // ─── summarizeNodes ────────────────────────────────────────────────────────

  describe('summarizeNodes', () => {
    function makeNodeContext(
      node: { id: Uuid },
      episode: EpisodicNode,
      previousEpisodes: EpisodicNode[] = [],
    ) {
      return new Map([
        [
          node.id,
          {
            episode,
            previousEpisodes,
            chunks: [episode.content],
            sourceChunkIndices: new Set([0]),
          },
        ],
      ]);
    }

    it('invokes structured output and applies returned summaries to canonical nodes', async () => {
      const node = KgNodeFactory.createEntityNode({ name: 'Alice' });
      mockRunnable.invoke.mockResolvedValue({
        summaries: [{ name: node.name, summary: 'Alice is an engineer' }],
      });

      await service.summarizeNodes(
        mockModel,
        [node],
        [],
        undefined,
        makeNodeContext(node, baseEpisode),
      );

      expect(mockModel.withStructuredOutput).toHaveBeenCalled();
      expect(node.summary).toBe('Alice is an engineer');
    });

    it('does not invoke structured output when there are no nodes', async () => {
      await service.summarizeNodes(mockModel, [], [], undefined, new Map());

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
    });
  });

  // ─── fillEntityAttributes ──────────────────────────────────────────────────

  describe('fillEntityAttributes', () => {
    it('does nothing when entityTypes is undefined', async () => {
      const node = KgNodeFactory.createEntityNode({ name: 'Alice' });

      await service.fillEntityAttributes(
        mockModel,
        [node],
        [],
        undefined,
        new Map([
          [
            node.id,
            {
              episode: baseEpisode,
              previousEpisodes: [],
              chunks: [baseEpisode.content],
              sourceChunkIndices: new Set([0]),
            },
          ],
        ]),
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
    });

    it('merges LLM-returned attributes onto the node', async () => {
      const entityTypes = {
        Person: {
          description: 'A human individual',
          schema: z.object({ age: z.number().optional() }),
        },
      };
      const node = KgNodeFactory.createEntityNode({
        name: 'Alice',
        labels: ['Entity', 'Person'],
      });
      const edge = KgEdgeFactory.createEntityEdge({
        sourceNodeId: node.id,
        targetNodeId: u('other'),
        name: 'KNOWS',
        fact: 'Alice knows Bob',
      });
      mockRunnable.invoke.mockResolvedValue({ age: 30 });

      await service.fillEntityAttributes(
        mockModel,
        [node],
        [edge],
        entityTypes,
        new Map([
          [
            node.id,
            {
              episode: baseEpisode,
              previousEpisodes: [],
              chunks: [baseEpisode.content],
              sourceChunkIndices: new Set([0]),
            },
          ],
        ]),
      );

      expect(node.attributes).toEqual({ age: 30 });
    });

    it('skips nodes whose label is not in entityTypes', async () => {
      const entityTypes = {
        Person: { description: 'A human individual', schema: z.object({}) },
      };
      const node = KgNodeFactory.createEntityNode({
        name: 'Acme',
        labels: ['Entity', 'Organization'],
      });

      await service.fillEntityAttributes(
        mockModel,
        [node],
        [],
        entityTypes,
        new Map([
          [
            node.id,
            {
              episode: baseEpisode,
              previousEpisodes: [],
              chunks: [baseEpisode.content],
              sourceChunkIndices: new Set([0]),
            },
          ],
        ]),
      );

      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
    });

    it('throws when a typed node is missing from nodeContext (invariant violation)', async () => {
      const entityTypes = {
        Person: { description: 'A human individual', schema: z.object({}) },
      };
      const node = KgNodeFactory.createEntityNode({
        name: 'Alice',
        labels: ['Entity', 'Person'],
      });

      await expect(
        service.fillEntityAttributes(mockModel, [node], [], entityTypes, new Map()),
      ).rejects.toThrow();
      expect(mockModel.withStructuredOutput).not.toHaveBeenCalled();
    });
  });
});
