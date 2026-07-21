import { z } from 'zod';

import { KgNodeFactory, u } from '@/test/factories';

import type { EntityCorefDescriptor } from '../extraction';
import type { NodeLabel, NodeLabels, RelationshipType } from '../types';
import { NodeLabelSchema, RelationshipTypeSchema } from '../types';
import {
  buildEdgeCorefCandidates,
  edgeTypeKey,
  getApplicableEdgeTypes,
  getEffectiveTypeMappings,
  recomputeCanonicalNodesByCanonicalIdMap,
  recomputeCorefByCanonicalIdMap,
} from './episode-utils';
import type { BatchState, EdgeTypeMap, EdgeTypeMappings, EpisodeWorkItem } from './types';

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

// ─── Canonical projection helpers (recompute*) ──────────────────────────────
// These mutate the shared items/batch pipeline state; the service specs only
// verify wiring through the public API.

function makeWorkItem(overrides: Partial<EpisodeWorkItem> = {}): EpisodeWorkItem {
  return {
    episode: KgNodeFactory.createEpisodicNode(),
    chunks: ['chunk'],
    prevEpisodes: [],
    chunkIndicesByExtractedId: new Map(),
    corefByExtractedId: new Map(),
    unresolvedReferences: [],
    nodeResolution: {
      newNodes: [],
      nodesMatchedToPreexistingNodes: [],
      preexistingCandidates: [],
    },
    canonicalNodes: [],
    committedCorefBindings: [],
    selfLoopFactsForEnrichment: [],
    edgeResolution: { resolvedEdges: [], invalidatedEdges: [], newEdges: [] },
    episodicEdges: [],
    ...overrides,
  };
}

function makeBatch(overrides: Partial<BatchState> = {}): BatchState {
  return {
    canonicalIdByNodeId: new Map(),
    allKnownNodesById: new Map(),
    preexistingGraphNodeIds: new Set(),
    chunkSources: new Map(),
    canonicalNodes: [],
    corefByCanonicalId: new Map(),
    ...overrides,
  };
}

function descriptor(
  identifyingDescription: string,
  ...aliases: string[]
): EntityCorefDescriptor {
  return { identifyingDescription, aliases, referredToAsPronouns: [] };
}

describe('recomputeCorefByCanonicalIdMap', () => {
  it('merges descriptors collapsed onto one canonical id: alias union, first description wins', () => {
    const clark = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const captain = KgNodeFactory.createEntityNode({ name: 'captain' });
    const item = makeWorkItem({
      corefByExtractedId: new Map([
        [clark.id, descriptor('the expedition captain', 'Captain Clark')],
        [captain.id, descriptor('ship captain', 'the captain', 'Captain Clark')],
      ]),
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[captain.id, clark.id]]),
    });

    recomputeCorefByCanonicalIdMap([item], batch);

    expect([...batch.corefByCanonicalId.keys()]).toEqual([clark.id]);
    const merged = batch.corefByCanonicalId.get(clark.id)!;
    expect(merged.identifyingDescription).toBe('the expedition captain');
    expect(merged.aliases).toEqual(['Captain Clark', 'the captain']);
  });

  it('unions observed pronouns onto the canonical id, keeping a conflicting union as-is', () => {
    const clark = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const captain = KgNodeFactory.createEntityNode({ name: 'captain' });
    const item = makeWorkItem({
      corefByExtractedId: new Map([
        [
          clark.id,
          {
            identifyingDescription: 'the expedition captain',
            aliases: [],
            referredToAsPronouns: ['he'],
          },
        ],
        [
          captain.id,
          {
            identifyingDescription: 'ship captain',
            aliases: [],
            referredToAsPronouns: ['she', 'he'],
          },
        ],
      ]),
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[captain.id, clark.id]]),
    });

    recomputeCorefByCanonicalIdMap([item], batch);

    const merged = batch.corefByCanonicalId.get(clark.id)!;
    expect(merged.referredToAsPronouns).toEqual(['he', 'she']);
  });

  it('keys a descriptor by the extracted id itself when the id map has no entry', () => {
    const node = KgNodeFactory.createEntityNode({ name: 'Alice' });
    const item = makeWorkItem({
      corefByExtractedId: new Map([[node.id, descriptor('a botanist', 'Alice')]]),
    });
    const batch = makeBatch();

    recomputeCorefByCanonicalIdMap([item], batch);

    expect([...batch.corefByCanonicalId.keys()]).toEqual([node.id]);
  });

  it('merges descriptors for the same canonical id across items', () => {
    const canonical = KgNodeFactory.createEntityNode({ name: 'Alice' });
    const aliasNode = KgNodeFactory.createEntityNode({ name: 'Alicia' });
    const item1 = makeWorkItem({
      corefByExtractedId: new Map([[canonical.id, descriptor('a botanist', 'Alice')]]),
    });
    const item2 = makeWorkItem({
      corefByExtractedId: new Map([[aliasNode.id, descriptor('the botanist', 'Alicia')]]),
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[aliasNode.id, canonical.id]]),
    });

    recomputeCorefByCanonicalIdMap([item1, item2], batch);

    expect(batch.corefByCanonicalId.size).toBe(1);
    const merged = batch.corefByCanonicalId.get(canonical.id)!;
    expect(merged.identifyingDescription).toBe('a botanist');
    expect(merged.aliases).toEqual(['Alice', 'Alicia']);
  });

  it('replaces the previous map instead of merging into it', () => {
    const stale = KgNodeFactory.createEntityNode({ name: 'Stale' });
    const batch = makeBatch({
      corefByCanonicalId: new Map([[stale.id, descriptor('stale entry')]]),
    });

    recomputeCorefByCanonicalIdMap([makeWorkItem()], batch);

    expect(batch.corefByCanonicalId.size).toBe(0);
  });

  it('leaves the source descriptors untouched when merging', () => {
    const clark = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const captain = KgNodeFactory.createEntityNode({ name: 'captain' });
    const clarkDescriptor = descriptor('the expedition captain', 'Captain Clark');
    const captainDescriptor = descriptor('ship captain', 'the captain');
    const item = makeWorkItem({
      corefByExtractedId: new Map([
        [clark.id, clarkDescriptor],
        [captain.id, captainDescriptor],
      ]),
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[captain.id, clark.id]]),
    });

    recomputeCorefByCanonicalIdMap([item], batch);

    expect(clarkDescriptor.aliases).toEqual(['Captain Clark']);
    expect(captainDescriptor.aliases).toEqual(['the captain']);
  });
});

describe('recomputeCanonicalNodesByCanonicalIdMap', () => {
  it('drops an episode-created node the id map redirects', () => {
    const clark = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const captain = KgNodeFactory.createEntityNode({ name: 'captain' });
    const item = makeWorkItem({
      nodeResolution: {
        newNodes: [captain, clark],
        nodesMatchedToPreexistingNodes: [],
        preexistingCandidates: [],
      },
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[captain.id, clark.id]]),
      allKnownNodesById: new Map([
        [captain.id, captain],
        [clark.id, clark],
      ]),
    });

    recomputeCanonicalNodesByCanonicalIdMap([item], batch);

    expect(item.canonicalNodes).toEqual([clark]);
    expect(batch.canonicalNodes).toEqual([clark]);
  });

  it('follows a matched preexisting node through the id map when it was merged onward', () => {
    const first = KgNodeFactory.createEntityNode({ name: 'A. Clark' });
    const merged = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const item = makeWorkItem({
      nodeResolution: {
        newNodes: [],
        nodesMatchedToPreexistingNodes: [
          { extractedId: u('extracted'), preexistingNodeId: first.id },
        ],
        preexistingCandidates: [first],
      },
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[first.id, merged.id]]),
      allKnownNodesById: new Map([
        [first.id, first],
        [merged.id, merged],
      ]),
    });

    recomputeCanonicalNodesByCanonicalIdMap([item], batch);

    expect(item.canonicalNodes).toHaveLength(1);
    expect(item.canonicalNodes[0]).toBe(merged);
  });

  it('drops a matched preexisting node whose current id is unknown to the batch', () => {
    const unknown = KgNodeFactory.createEntityNode({ name: 'Ghost' });
    const item = makeWorkItem({
      nodeResolution: {
        newNodes: [],
        nodesMatchedToPreexistingNodes: [
          { extractedId: u('extracted'), preexistingNodeId: unknown.id },
        ],
        preexistingCandidates: [],
      },
    });
    const batch = makeBatch();

    recomputeCanonicalNodesByCanonicalIdMap([item], batch);

    expect(item.canonicalNodes).toHaveLength(0);
    expect(batch.canonicalNodes).toHaveLength(0);
  });

  it('dedupes two matches resolving to the same preexisting node', () => {
    const live = KgNodeFactory.createEntityNode({ name: 'Alice' });
    const item = makeWorkItem({
      nodeResolution: {
        newNodes: [],
        nodesMatchedToPreexistingNodes: [
          { extractedId: u('extracted-1'), preexistingNodeId: live.id },
          { extractedId: u('extracted-2'), preexistingNodeId: live.id },
        ],
        preexistingCandidates: [live],
      },
    });
    const batch = makeBatch({
      allKnownNodesById: new Map([[live.id, live]]),
    });

    recomputeCanonicalNodesByCanonicalIdMap([item], batch);

    expect(item.canonicalNodes).toEqual([live]);
  });

  it('unions items into batch.canonicalNodes without duplicates, sharing object refs', () => {
    const live = KgNodeFactory.createEntityNode({ name: 'Alice' });
    const makeMatchedItem = (extractedId: string) =>
      makeWorkItem({
        nodeResolution: {
          newNodes: [],
          nodesMatchedToPreexistingNodes: [
            { extractedId: u(extractedId), preexistingNodeId: live.id },
          ],
          preexistingCandidates: [live],
        },
      });
    const items = [makeMatchedItem('extracted-1'), makeMatchedItem('extracted-2')];
    const batch = makeBatch({
      allKnownNodesById: new Map([[live.id, live]]),
    });

    recomputeCanonicalNodesByCanonicalIdMap(items, batch);

    expect(items[0].canonicalNodes[0]).toBe(live);
    expect(items[1].canonicalNodes[0]).toBe(live);
    expect(batch.canonicalNodes).toHaveLength(1);
    expect(batch.canonicalNodes[0]).toBe(live);
  });

  it('replaces stale canonicalNodes on both the item and the batch', () => {
    const stale = KgNodeFactory.createEntityNode({ name: 'Stale' });
    const item = makeWorkItem({ canonicalNodes: [stale] });
    const batch = makeBatch({ canonicalNodes: [stale] });

    recomputeCanonicalNodesByCanonicalIdMap([item], batch);

    expect(item.canonicalNodes).toHaveLength(0);
    expect(batch.canonicalNodes).toHaveLength(0);
  });
});

describe('buildEdgeCorefCandidates', () => {
  it('builds a scoped candidate from a canonical node with a descriptor and chunk indices', () => {
    const node = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const item = makeWorkItem({
      canonicalNodes: [node],
      chunkIndicesByExtractedId: new Map([[node.id, new Set([2, 1])]]),
    });
    const batch = makeBatch({
      corefByCanonicalId: new Map([
        [node.id, descriptor('the expedition captain', 'the captain')],
      ]),
    });

    expect(buildEdgeCorefCandidates(item, batch)).toEqual([
      {
        name: node.name,
        identifyingDescription: 'the expedition captain',
        aliases: ['the captain'],
        referredToAsPronouns: [],
        introChunk: 1,
      },
    ]);
  });

  it('takes the min over indices unioned from merged-away extracted nodes', () => {
    const clark = KgNodeFactory.createEntityNode({ name: 'Captain Clark' });
    const captain = KgNodeFactory.createEntityNode({ name: 'captain' });
    const item = makeWorkItem({
      canonicalNodes: [clark],
      chunkIndicesByExtractedId: new Map([
        [clark.id, new Set([2])],
        [captain.id, new Set([0, 3])],
      ]),
    });
    const batch = makeBatch({
      canonicalIdByNodeId: new Map([[captain.id, clark.id]]),
      corefByCanonicalId: new Map([[clark.id, descriptor('the captain')]]),
    });

    const [candidate] = buildEdgeCorefCandidates(item, batch);

    expect(candidate.introChunk).toBe(0);
  });

  it('returns no candidates when the episode has no canonical nodes', () => {
    expect(buildEdgeCorefCandidates(makeWorkItem(), makeBatch())).toEqual([]);
  });

  it('throws when a canonical node has no descriptor', () => {
    const node = KgNodeFactory.createEntityNode({ name: 'Ghost' });
    const item = makeWorkItem({
      canonicalNodes: [node],
      chunkIndicesByExtractedId: new Map([[node.id, new Set([0])]]),
    });

    expect(() => buildEdgeCorefCandidates(item, makeBatch())).toThrow(Error);
  });

  it('throws when a canonical node has no originating chunk indices', () => {
    const node = KgNodeFactory.createEntityNode({ name: 'Ghost' });
    const item = makeWorkItem({ canonicalNodes: [node] });
    const batch = makeBatch({
      corefByCanonicalId: new Map([[node.id, descriptor('a ghost')]]),
    });

    expect(() => buildEdgeCorefCandidates(item, batch)).toThrow(Error);
  });
});
