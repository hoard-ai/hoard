import { KG_REFERENCE_TIME, KG_TEST_GRAPH_ID, kgId } from '@/test/factories';

import { RelationshipTypeSchema } from '../types';
import {
  createEdgeDefaults,
  createEntityEdge,
  createEpisodicEdge,
  EdgeBaseSchema,
  EntityEdgeSchema,
  EpisodicEdgeSchema,
} from './edge.types';

describe('edge.types', () => {
  describe('createEdgeDefaults', () => {
    it('should create defaults with a valid id', () => {
      const defaults = createEdgeDefaults();
      expect(defaults.id).toBeDefined();
      expect(defaults.id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
      );
    });

    it('should not include graphId or node ids in defaults', () => {
      const defaults = createEdgeDefaults();
      expect('graphId' in defaults).toBe(false);
      expect('sourceNodeId' in defaults).toBe(false);
      expect('targetNodeId' in defaults).toBe(false);
    });

    it('should create defaults with a Date for createdAt', () => {
      const defaults = createEdgeDefaults();
      expect(defaults.createdAt).toBeInstanceOf(Date);
    });
  });

  describe('EdgeBaseSchema', () => {
    it('should accept a valid edge base', () => {
      const edge = {
        id: kgId(),
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId: kgId(),
        targetNodeId: kgId(),
        createdAt: new Date(),
      };
      expect(() => EdgeBaseSchema.parse(edge)).not.toThrow();
    });

    it('should reject invalid source id', () => {
      const edge = {
        id: kgId(),
        graphId: 'group-1',
        sourceNodeId: 'not-a-uuid',
        targetNodeId: kgId(),
        createdAt: new Date(),
      };
      expect(() => EdgeBaseSchema.parse(edge)).toThrow();
    });

    it('should reject invalid target id', () => {
      const edge = {
        id: kgId(),
        graphId: 'group-1',
        sourceNodeId: kgId(),
        targetNodeId: 'not-a-uuid',
        createdAt: new Date(),
      };
      expect(() => EdgeBaseSchema.parse(edge)).toThrow();
    });

    it('should reject empty graphId', () => {
      const edge = {
        id: kgId(),
        graphId: '',
        sourceNodeId: kgId(),
        targetNodeId: kgId(),
        createdAt: new Date(),
      };
      expect(() => EdgeBaseSchema.parse(edge)).toThrow();
    });
  });
});

describe('EntityEdge', () => {
  const sourceNodeId = kgId();
  const targetNodeId = kgId();

  describe('createEntityEdge', () => {
    it('should create with correct defaults', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(edge.name).toBe('KNOWS');
      expect(edge.sourceNodeId).toBe(sourceNodeId);
      expect(edge.targetNodeId).toBe(targetNodeId);
      expect(edge.fact).toBe('A knows B');
      expect(edge.factEmbedding).toBeNull();
      expect(edge.episodes).toEqual([]);
      expect(edge.expiredAt).toBeNull();
      expect(edge.validAt).toBeNull();
      expect(edge.invalidAt).toBeNull();
      expect(edge.attributes).toEqual({});
      expect(edge.id).toBeDefined();
      expect(edge.createdAt).toBeInstanceOf(Date);
    });

    it('should allow overriding fact', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'Person A knows Person B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(edge.fact).toBe('Person A knows Person B');
    });

    it('should allow setting factEmbedding', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
        factEmbedding: [0.1, 0.2, 0.3],
      });
      expect(edge.factEmbedding).toEqual([0.1, 0.2, 0.3]);
    });

    it('should allow setting dates', () => {
      const date = KG_REFERENCE_TIME;
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
        validAt: date,
        expiredAt: date,
        invalidAt: date,
      });
      expect(edge.validAt).toBe(date);
      expect(edge.expiredAt).toBe(date);
      expect(edge.invalidAt).toBe(date);
    });
  });

  describe('EntityEdgeSchema', () => {
    it('should accept valid entity edge', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() => EntityEdgeSchema.parse(edge)).not.toThrow();
    });

    it('should reject empty name', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() => EntityEdgeSchema.parse({ ...edge, name: '' })).toThrow();
    });

    it('should reject empty graphId', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() => EntityEdgeSchema.parse({ ...edge, graphId: '' })).toThrow();
    });

    it('should accept null dates', () => {
      const edge = createEntityEdge({
        name: RelationshipTypeSchema.parse('KNOWS'),
        fact: 'A knows B',
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
        validAt: null,
        expiredAt: null,
        invalidAt: null,
      });
      expect(() => EntityEdgeSchema.parse(edge)).not.toThrow();
    });
  });
});

describe('EpisodicEdge', () => {
  const sourceNodeId = kgId();
  const targetNodeId = kgId();

  describe('createEpisodicEdge', () => {
    it('should create with correct defaults', () => {
      const edge = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(edge.sourceNodeId).toBe(sourceNodeId);
      expect(edge.targetNodeId).toBe(targetNodeId);
      expect(edge.id).toBeDefined();
      expect(edge.createdAt).toBeInstanceOf(Date);
      expect(edge.graphId).toBe(KG_TEST_GRAPH_ID);
    });

    it('should generate unique ids', () => {
      const edge1 = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      const edge2 = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(edge1.id).not.toBe(edge2.id);
    });
  });

  describe('EpisodicEdgeSchema', () => {
    it('should accept valid episodic edge', () => {
      const edge = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() => EpisodicEdgeSchema.parse(edge)).not.toThrow();
    });

    it('should reject empty graphId', () => {
      const edge = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() => EpisodicEdgeSchema.parse({ ...edge, graphId: '' })).toThrow();
    });

    it('should reject invalid source id', () => {
      const edge = createEpisodicEdge({
        graphId: KG_TEST_GRAPH_ID,
        sourceNodeId,
        targetNodeId,
      });
      expect(() =>
        EpisodicEdgeSchema.parse({ ...edge, sourceNodeId: 'not-a-uuid' }),
      ).toThrow();
    });
  });
});
