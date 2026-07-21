import { z } from 'zod';

import type { Uuid } from '@/common/schemas';
import { UuidSchema } from '@/common/schemas';

import type { EntityEdge } from '../../models';
import { EntityEdgeSchema, EntityNodeSchema } from '../../models';

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
  // preexisting graph edges). Attribute extraction runs only on these to avoid
  // overwriting prior values when a preexisting edge is matched as a duplicate.
  newEdges: z.array(EntityEdgeSchema),
});

export const NodeResolutionResultSchema = z.object({
  // Extracted nodes with no live-graph match: the node objects this episode created.
  newNodes: z.array(EntityNodeSchema),
  // Extracted nodes matched to graph rows that predate this batch.
  nodesMatchedToPreexistingNodes: z.array(
    z.object({ extractedId: UuidSchema, preexistingNodeId: UuidSchema }),
  ),
  // Surfaced so the orchestrator can seed cross-batch dedup and the
  // preexisting-id set.
  preexistingCandidates: z.array(EntityNodeSchema),
});

// Types

export type EdgeResolutionResult = z.infer<typeof EdgeResolutionResultSchema>;
export type NodeResolutionResult = z.infer<typeof NodeResolutionResultSchema>;

export type DedupeEdgesResult = {
  // Preexisting graph edges that an extracted edge duplicated; episodes appended,
  // re-saved as-is (not enriched - they keep their prior attributes/bounds).
  matchedPreexistingEdges: EntityEdge[];
  // Freshly extracted edges with no duplicate in the graph. These
  // need enrichment (timestamps + attributes) and invalidation.
  newEdges: EntityEdge[];
  // Per new edge: the preexisting graph edges it contradicts, carried to the
  // invalidation stage (which runs after timestamps are filled).
  contradictionsByNewEdgeId: Map<Uuid, EntityEdge[]>;
};
