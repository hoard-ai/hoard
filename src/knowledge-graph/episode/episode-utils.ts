import { Uuid } from '@/common';

import { NodeEpisodeContext } from '../extraction';
import { EntityNode, EpisodicNode } from '../models';
import { NodeLabel, NodeLabels, NodeLabelSchema, RelationshipType } from '../types';
import { EdgeTypeMap, EdgeTypeMappings } from './types';

/**
 * Returns the subset of `edgeTypes` that are valid for the given source/target
 * label combination, as determined by `edgeTypeMap`.
 *
 * `edgeTypeMap` keys are `"SourceLabel,TargetLabel"` strings. For each
 * combination of source and target labels, the map yields edge type names whose
 * definitions are then looked up in `edgeTypes`. Duplicates are deduplicated
 * (first occurrence wins).
 *
 * @example
 * // sourceLabels: ['Person'], targetLabels: ['Company']
 * // edgeTypeMap:  { 'Person,Company': ['WORKS_AT', 'FOUNDED'] }
 * // edgeTypes:    { WORKS_AT: { description: '...', schema: ... }, FOUNDED: { ... } }
 * // → { WORKS_AT: { description: '...', schema: ... }, FOUNDED: { ... } }
 */
export function getApplicableEdgeTypes(
  sourceLabels: NodeLabels,
  targetLabels: NodeLabels,
  edgeTypes: EdgeTypeMap,
  edgeTypeMappings: EdgeTypeMappings,
): EdgeTypeMap {
  const result: EdgeTypeMap = {};

  for (const src of sourceLabels) {
    for (const tgt of targetLabels) {
      const key: [NodeLabel, NodeLabel] = [src, tgt];

      for (const typeName of edgeTypeMappings.get(key) ?? []) {
        const typeDef = edgeTypes[typeName];
        if (typeDef && !(typeName in result)) result[typeName] = typeDef;
      }
    }
  }
  return result;
}

export function getEffectiveTypeMappings(
  edgeTypeMappings?: EdgeTypeMappings,
  edgeTypes?: EdgeTypeMap,
): EdgeTypeMappings | undefined {
  let effectiveEdgeTypeMappings = edgeTypeMappings;

  if (!edgeTypeMappings && edgeTypes) {
    const defaultKey: [NodeLabel, NodeLabel] = [
      NodeLabelSchema.parse('Entity'),
      NodeLabelSchema.parse('Entity'),
    ];
    effectiveEdgeTypeMappings = new Map();

    effectiveEdgeTypeMappings.set(
      defaultKey,
      Object.keys(edgeTypes) as RelationshipType[],
    );
  }
  return effectiveEdgeTypeMappings;
}

/**
 * Maps each canonical node to its episode context (chunks + the chunk indices
 * it was extracted from) for the attribute/summary helpers. Within an episode,
 * indices from nodes merged into one canonical id are unioned; across episodes
 * the first episode to reference a canonical node wins.
 */
export function buildNodeContext(
  canonicalNodesPerEpisode: EntityNode[][],
  chunkIndicesByNodeIdPerEpisode: Map<Uuid, Set<number>>[],
  canonicalIdByNodeId: Map<Uuid, Uuid>,
  episodicNodes: EpisodicNode[],
  prevEpisodesPerEpisode: EpisodicNode[][],
  chunksPerEpisode: string[][],
): NodeEpisodeContext {
  const nodeContext: NodeEpisodeContext = new Map();

  canonicalNodesPerEpisode.forEach((nodes, i) => {
    // Remap this episode's extracted-node chunk indices onto canonical ids so
    // each canonical node's episode text covers every chunk it was extracted from.
    const chunkIndicesByCanonicalId = new Map<Uuid, Set<number>>();

    for (const [extractedId, idxs] of chunkIndicesByNodeIdPerEpisode[i]) {
      const canonicalId = canonicalIdByNodeId.get(extractedId) ?? extractedId;
      let indicesSet = chunkIndicesByCanonicalId.get(canonicalId);

      if (!indicesSet) {
        indicesSet = new Set();
        chunkIndicesByCanonicalId.set(canonicalId, indicesSet);
      }
      for (const idx of idxs) indicesSet.add(idx);
    }

    for (const n of nodes) {
      // TODO: first-episode-wins. A node mentioned in several batch episodes
      // uses only the first episode's chunk text (its facts still arrive via
      // edges). Full fix = multi-episode context (chunk text union + per-episode
      // referenceTime/previousEpisodes/summary grouping); gate on eval.
      if (nodeContext.has(n.id)) continue;

      // Every canonical node traces back to an extracted node of this episode
      const sourceChunkIndices = chunkIndicesByCanonicalId.get(n.id);
      if (!sourceChunkIndices) {
        throw new Error(
          `nodeContext: canonical node ${n.id} has no originating chunk indices`,
        );
      }
      nodeContext.set(n.id, {
        episode: episodicNodes[i],
        previousEpisodes: prevEpisodesPerEpisode[i],
        chunks: chunksPerEpisode[i],
        sourceChunkIndices,
      });
    }
  });
  return nodeContext;
}
