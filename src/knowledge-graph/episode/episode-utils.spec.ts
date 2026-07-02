import { z } from 'zod';

import {
  NodeLabel,
  NodeLabels,
  NodeLabelSchema,
  RelationshipType,
  RelationshipTypeSchema,
} from '../types';
import {
  edgeTypeKey,
  getApplicableEdgeTypes,
  getEffectiveTypeMappings,
} from './episode-utils';
import { EdgeTypeMap, EdgeTypeMappings } from './types';

const label = (s: string): NodeLabel => NodeLabelSchema.parse(s);
const rel = (s: string): RelationshipType => RelationshipTypeSchema.parse(s);
const labels = (...ss: string[]): NodeLabels => ss.map(label);

const attr = z.object({ since: z.string() });

const edgeTypes: EdgeTypeMap = {
  [rel('WORKS_AT')]: { description: 'employment', schema: attr },
  [rel('FOUNDED')]: { description: 'founding', schema: attr },
};

describe('edgeTypeKey', () => {
  it('joins a label pair into a canonical comma-separated key', () => {
    expect(edgeTypeKey(label('Person'), label('Company'))).toBe('Person,Company');
  });
});

describe('getApplicableEdgeTypes', () => {
  it('resolves types for a matching pair despite tuple-keyed Map lookup', () => {
    // A native Map keyed by tuples cannot be read with a freshly built tuple;
    // the source/target labels here are distinct objects from the map's key.
    const mappings: EdgeTypeMappings = new Map([
      [
        [label('Person'), label('Company')],
        [rel('WORKS_AT'), rel('FOUNDED')],
      ],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Person'),
      labels('Company'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result)).toHaveLength(2);
    // The full definition (schema + description) is carried through, not a
    // key-only stub - enrichEdges reads `applicable[edge.name].schema`.
    expect(result[rel('WORKS_AT')].schema).toBe(attr);
    expect(result[rel('WORKS_AT')].description).toBe('employment');
    expect(result[rel('FOUNDED')].schema).toBe(attr);
    expect(result[rel('FOUNDED')].description).toBe('founding');
  });

  it('returns an empty map when no label pair matches', () => {
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('WORKS_AT')]],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Person'),
      labels('Person'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result)).toHaveLength(0);
  });

  it('deduplicates type names shared across matching pairs (first wins)', () => {
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('WORKS_AT')]],
      [[label('Person'), label('Org')], [rel('WORKS_AT')]],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Person'),
      labels('Company', 'Org'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result)).toHaveLength(1);
    expect(result[rel('WORKS_AT')].schema).toBe(attr);
    expect(result[rel('WORKS_AT')].description).toBe('employment');
  });

  it("matches on any of a node's labels (multi-label source)", () => {
    // Entity nodes carry ['Entity', <SpecificType>]; the pair matches on Person,
    // not the leading Entity label.
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('WORKS_AT')]],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Entity', 'Person'),
      labels('Company'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result)).toEqual(['WORKS_AT']);
    expect(result[rel('WORKS_AT')].schema).toBe(attr);
  });

  it('unions distinct types across matching label combinations', () => {
    // Person,Company -> WORKS_AT and Entity,Entity -> FOUNDED both fire for
    // nodes labelled ['Entity', <SpecificType>]; the result is their union.
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('WORKS_AT')]],
      [[label('Entity'), label('Entity')], [rel('FOUNDED')]],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Entity', 'Person'),
      labels('Entity', 'Company'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result).sort()).toEqual(['FOUNDED', 'WORKS_AT']);
    // Distinct descriptions confirm each pair resolved to its own definition.
    expect(result[rel('WORKS_AT')].description).toBe('employment');
    expect(result[rel('FOUNDED')].description).toBe('founding');
  });

  it('skips mapped type names that have no definition in edgeTypes', () => {
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('UNDEFINED_TYPE')]],
    ]);

    const result = getApplicableEdgeTypes(
      labels('Person'),
      labels('Company'),
      edgeTypes,
      mappings,
    );

    expect(Object.keys(result)).toHaveLength(0);
  });
});

describe('getEffectiveTypeMappings', () => {
  it('returns provided mappings unchanged', () => {
    const mappings: EdgeTypeMappings = new Map([
      [[label('Person'), label('Company')], [rel('WORKS_AT')]],
    ]);

    expect(getEffectiveTypeMappings(mappings, edgeTypes)).toBe(mappings);
  });

  it('returns undefined when neither mappings nor edgeTypes are given', () => {
    expect(getEffectiveTypeMappings(undefined, undefined)).toBeUndefined();
  });

  it('derives an Entity,Entity default that round-trips through getApplicableEdgeTypes', () => {
    const effective = getEffectiveTypeMappings(undefined, edgeTypes);
    expect(effective).toBeInstanceOf(Map);

    const result = getApplicableEdgeTypes(
      labels('Entity'),
      labels('Entity'),
      edgeTypes,
      effective as EdgeTypeMappings,
    );

    expect(Object.keys(result)).toHaveLength(Object.keys(edgeTypes).length);
    expect(result[rel('WORKS_AT')].schema).toBe(attr);
    expect(result[rel('FOUNDED')].schema).toBe(attr);
  });
});
