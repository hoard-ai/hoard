import { z } from 'zod';

import { Uuid, UuidSchema } from '@/common/schemas';

import { EntityEdge, EntityEdgeSchema, EntityNodeSchema } from '../../models';

// Edge chunk provenance, keyed by edge id and qualified by origin episode, so a
// cross-episode-merged edge resolves against the chunks it actually came from.
// Entries are immutable (never unioned at merge); the future EpisodeChunk join
// remaps merged edge ids -> canonical via the dedup idMap.
export type EdgeChunkSources = Map<Uuid, { episodeIndex: number; indices: Set<number> }>;

// Schemas

export const EdgeResolutionResultSchema = z.object({
  resolvedEdges: z.array(EntityEdgeSchema),
  invalidatedEdges: z.array(EntityEdgeSchema),
  // Subset of resolvedEdges that were freshly extracted (not duplicates of
  // existing graph edges). Attribute extraction runs only on these to avoid
  // overwriting prior values when an existing edge is matched as a duplicate.
  newEdges: z.array(EntityEdgeSchema),
});

export const NodeResolutionResultSchema = z.object({
  resolvedNodes: z.array(EntityNodeSchema),
  idMap: z.map(UuidSchema, UuidSchema),
  duplicatePairs: z.array(z.object({ extractedId: UuidSchema, canonicalId: UuidSchema })),
  // Live-graph candidates collected during resolution, surfaced so the
  // orchestrator can seed cross-batch dedup and identify pre-existing nodes.
  candidates: z.array(EntityNodeSchema),
});

// Types

export type EdgeResolutionResult = z.infer<typeof EdgeResolutionResultSchema>;
export type NodeResolutionResult = z.infer<typeof NodeResolutionResultSchema>;

export type DedupeEdgesResult = {
  // Existing graph edges that an extracted edge duplicated; episodes appended,
  // re-saved as-is (not enriched - they keep their prior attributes/bounds).
  matchedExistingEdges: EntityEdge[];
  // Freshly extracted edges with no duplicate in the graph (= newEdges). These
  // need enrichment (timestamps + attributes) and invalidation.
  survivors: EntityEdge[];
  // Per survivor: the existing graph edges it contradicts, carried to the
  // invalidation stage (which runs after timestamps are filled).
  contradictionsBySurvivorId: Map<Uuid, EntityEdge[]>;
};
