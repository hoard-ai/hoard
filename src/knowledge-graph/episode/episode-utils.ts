import type { Uuid } from '@/common';

import type {
  CommittedCorefBinding,
  EntityCorefDescriptor,
  NodeEpisodeContext,
} from '../extraction';
import type { EntityNode, EpisodicNode } from '../models';
import type { ScopedCandidate } from '../prompts/coref-utils';
import type { NodeLabel, NodeLabels, RelationshipType } from '../types';
import { NodeLabelSchema } from '../types';
import type { BatchState, EdgeTypeMap, EdgeTypeMappings, EpisodeWorkItem } from './types';

export function edgeTypeKey(source: NodeLabel, target: NodeLabel): string {
  return `${source},${target}`;
}

/**
 * Returns the subset of `edgeTypes` that are valid for the given source/target
 * label combination, as determined by `edgeTypeMappings`.
 *
 * `edgeTypeMappings` is keyed by `[SourceLabel, TargetLabel]` tuples, matched
 * here via `edgeTypeKey`. For each combination of source and target labels, the
 * map yields edge type names whose definitions are then looked up in
 * `edgeTypes`. Duplicates are deduplicated (first occurrence wins).
 *
 * @example
 * // sourceLabels: ['Person'], targetLabels: ['Company']
 * // edgeTypeMappings: Map { ['Person','Company'] => ['WORKS_AT', 'FOUNDED'] }
 * // edgeTypes:        { WORKS_AT: { description: '...', schema: ... }, FOUNDED: { ... } }
 * // → { WORKS_AT: { description: '...', schema: ... }, FOUNDED: { ... } }
 */
export function getApplicableEdgeTypes(
  sourceLabels: NodeLabels,
  targetLabels: NodeLabels,
  edgeTypes: EdgeTypeMap,
  edgeTypeMappings: EdgeTypeMappings,
): EdgeTypeMap {
  const result: EdgeTypeMap = {};

  const namesByKey = new Map<string, RelationshipType[]>();
  for (const [[src, tgt], typeNames] of edgeTypeMappings) {
    namesByKey.set(edgeTypeKey(src, tgt), typeNames);
  }

  for (const src of sourceLabels) {
    for (const tgt of targetLabels) {
      for (const typeName of namesByKey.get(edgeTypeKey(src, tgt)) ?? []) {
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
 * Remaps a per-episode `extractedId -> chunk indices` map onto canonical ids,
 * unioning the indices of nodes merged into one canonical id. Shared by the node
 * context builder and the edge phase's coref candidate scoping.
 */
export function remapChunkIndicesToCanonical(
  chunkIndicesByExtractedId: Map<Uuid, Set<number>>,
  canonicalIdByNodeId: Map<Uuid, Uuid>,
): Map<Uuid, Set<number>> {
  const byCanonicalId = new Map<Uuid, Set<number>>();

  for (const [extractedId, idxs] of chunkIndicesByExtractedId) {
    const canonicalId = canonicalIdByNodeId.get(extractedId) ?? extractedId;
    let indicesSet = byCanonicalId.get(canonicalId);

    if (!indicesSet) {
      indicesSet = new Set();
      byCanonicalId.set(canonicalId, indicesSet);
    }
    for (const idx of idxs) indicesSet.add(idx);
  }
  return byCanonicalId;
}

/**
 * Maps each canonical node to its episode context (chunks + the chunk indices
 * it was extracted from, plus the episode's committed coref bindings) for the
 * attribute/summary helpers. Within an episode, indices from nodes merged into one
 * canonical id are unioned; across episodes the first episode to reference a
 * canonical node wins.
 */
export function buildNodeContext(
  canonicalNodesPerEpisode: EntityNode[][],
  chunkIndicesByExtractedIdPerEpisode: Map<Uuid, Set<number>>[],
  canonicalIdByNodeId: Map<Uuid, Uuid>,
  episodicNodes: EpisodicNode[],
  prevEpisodesPerEpisode: EpisodicNode[][],
  chunksPerEpisode: string[][],
  committedCorefBindingsPerEpisode: CommittedCorefBinding[][],
): NodeEpisodeContext {
  const nodeContext: NodeEpisodeContext = new Map();

  canonicalNodesPerEpisode.forEach((nodes, i) => {
    const chunkIndicesByCanonicalId = remapChunkIndicesToCanonical(
      chunkIndicesByExtractedIdPerEpisode[i],
      canonicalIdByNodeId,
    );

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
        committedCorefBindings: committedCorefBindingsPerEpisode[i],
      });
    }
  });
  return nodeContext;
}

/**
 * Coref candidate antecedents for this episode's edge extraction: canonical
 * nodes that carry a descriptor, each tagged with the chunk it was introduced
 * in (min of its remapped chunk indices) so extract-edges can apply the prefix.
 *
 * Reads:
 *   it.canonicalNodes
 *   it.chunkIndicesByExtractedId
 *   batch.canonicalIdByNodeId
 *   batch.corefByCanonicalId
 * Writes:
 *   nothing (returns the candidate list)
 */
export function buildEdgeCorefCandidates(
  it: EpisodeWorkItem,
  batch: BatchState,
): ScopedCandidate[] {
  const introChunkByCanonicalId = new Map<Uuid, number>();
  for (const [id, indices] of remapChunkIndicesToCanonical(
    it.chunkIndicesByExtractedId,
    batch.canonicalIdByNodeId,
  )) {
    introChunkByCanonicalId.set(id, Math.min(...indices));
  }

  const candidates: ScopedCandidate[] = [];
  for (const node of it.canonicalNodes) {
    const descriptor = batch.corefByCanonicalId.get(node.id);
    const introChunk = introChunkByCanonicalId.get(node.id);
    // With coref on, every canonical node traces back to a this-episode extracted
    // node, so it must carry a descriptor and originating chunk indices (same
    // invariant buildNodeContext enforces). A miss is a bookkeeping bug.
    if (!descriptor || introChunk === undefined) {
      throw new Error(
        `buildEdgeCorefCandidates: canonical node ${node.id} is missing its coref descriptor or originating chunk indices`,
      );
    }
    candidates.push({
      name: node.name,
      identifyingDescription: descriptor.identifyingDescription,
      aliases: descriptor.aliases,
      referredToAsPronouns: descriptor.referredToAsPronouns,
      introChunk,
    });
  }
  return candidates;
}

/**
 * Merges each extracted node's coref descriptor onto its canonical id (union
 * aliases and observed pronouns, keep first identifyingDescription). A pronoun
 * union that conflicts ("he" + "she") is kept as-is: the conflict is signal of
 * a bad extraction or merge, and consumers must see the ambiguity rather than
 * a false certainty.
 *
 * Reads:
 *   items[].corefByExtractedId
 *   batch.canonicalIdByNodeId
 * Writes:
 *   batch.corefByCanonicalId (replaced)
 */
export function recomputeCorefByCanonicalIdMap(
  items: EpisodeWorkItem[],
  batch: BatchState,
): void {
  const merged = new Map<Uuid, EntityCorefDescriptor>();
  for (const it of items) {
    for (const [extractedId, descriptor] of it.corefByExtractedId) {
      const canonicalId = batch.canonicalIdByNodeId.get(extractedId) ?? extractedId;
      const existing = merged.get(canonicalId);
      if (existing) {
        for (const alias of descriptor.aliases) {
          if (!existing.aliases.includes(alias)) existing.aliases.push(alias);
        }
        for (const pronoun of descriptor.referredToAsPronouns) {
          if (!existing.referredToAsPronouns.includes(pronoun)) {
            existing.referredToAsPronouns.push(pronoun);
          }
        }
      } else {
        merged.set(canonicalId, {
          identifyingDescription: descriptor.identifyingDescription,
          aliases: [...descriptor.aliases],
          referredToAsPronouns: [...descriptor.referredToAsPronouns],
        });
      }
    }
  }
  batch.corefByCanonicalId = merged;
}

/**
 * Recomputes which node objects each episode is about, under the current id
 * map. Re-run whenever the map changes, so merged-away duplicate nodes drop out
 * (e.g. "captain" merged into "Captain Clark").
 *
 * Reads:
 *   items[].nodeResolution.newNodes
 *   items[].nodeResolution.nodesMatchedToPreexistingNodes
 *   batch.canonicalIdByNodeId
 *   batch.allKnownNodesById
 * Writes:
 *   items[].canonicalNodes (replaced)
 *   batch.canonicalNodes (replaced)
 */
export function recomputeCanonicalNodesByCanonicalIdMap(
  items: EpisodeWorkItem[],
  batch: BatchState,
): void {
  items.forEach((it) => {
    // Of the nodes this episode created, those the id map does not redirect.
    const survivingOwnNodes = it.nodeResolution.newNodes.filter(
      (n) => (batch.canonicalIdByNodeId.get(n.id) ?? n.id) === n.id,
    );
    // The preexisting nodes this episode matched onto, each followed through the
    // id map in case it was itself merged onward after the match.
    const matchedPreexisting = it.nodeResolution.nodesMatchedToPreexistingNodes
      .map((p) => {
        const current =
          batch.canonicalIdByNodeId.get(p.preexistingNodeId) ?? p.preexistingNodeId;
        return batch.allKnownNodesById.get(current);
      })
      .filter((n): n is EntityNode => n !== undefined);

    const seen = new Set<Uuid>();
    it.canonicalNodes = [...survivingOwnNodes, ...matchedPreexisting].filter((n) => {
      if (seen.has(n.id)) return false;
      seen.add(n.id);
      return true;
    });
  });

  batch.canonicalNodes = [
    ...new Map(items.flatMap((it) => it.canonicalNodes).map((n) => [n.id, n])).values(),
  ];
}
