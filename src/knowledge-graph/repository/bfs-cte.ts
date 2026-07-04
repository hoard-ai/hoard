import { Prisma } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';

/**
 * Builds the `WITH RECURSIVE bfs(id, kind, depth, visited) AS (...)` CTE
 * shared by EntityNodeRepository.searchByBfs and EntityEdgeRepository.searchByBfs.
 *
 * The CTE seeds from `originNodeIds` resolved against both entity_nodes and
 * episodic_nodes (tagging each row's `kind`), then expands one entity-graph
 * hop per step. Episodic seeds expand via episodic_edges (MENTIONS) to entity
 * once at depth 1; from there it's a pure entity-graph traversal.
 *
 * Returned fragment includes the leading `WITH RECURSIVE`, so it must be the
 * first SQL in the statement. Callers append their outer SELECT after it:
 *
 *   const cte = buildBfsCte(originNodeIds, graphIds, depth);
 *   await prisma.$queryRaw`
 *     ${cte}
 *     SELECT ... FROM bfs b JOIN entity_nodes n ON n.id = b.id WHERE ...
 *   `;
 *
 * Shape constraints worth knowing: Postgres requires exactly one non-recursive
 * term + one recursive term separated by a single UNION [ALL], and the
 * recursive term may reference the CTE at most once (error 42P19). The two
 * seed arms are collapsed into a subquery anchor; the two expansion arms are
 * merged through a LATERAL subquery so `bfs` is referenced exactly once.
 */
export function buildBfsCte(
  originNodeIds: Uuid[],
  graphIds: Uuid[],
  depth: number,
): Prisma.Sql {
  if (!Number.isInteger(depth) || depth < 1) {
    throw new Error(
      `buildBfsCte: depth must be a positive integer, got ${String(depth)}`,
    );
  }
  return Prisma.sql`
    WITH RECURSIVE bfs(id, kind, depth, visited) AS (
      SELECT seeds.id, seeds.kind, 0, ARRAY[seeds.id]
      FROM (
        SELECT id, 'entity'::text AS kind
        FROM entity_nodes
        WHERE id = ANY(${originNodeIds}::uuid[]) AND graph_id = ANY(${graphIds}::uuid[])
        UNION ALL
        SELECT id, 'episodic'::text AS kind
        FROM episodic_nodes
        WHERE id = ANY(${originNodeIds}::uuid[]) AND graph_id = ANY(${graphIds}::uuid[])
      ) seeds

      UNION ALL

      SELECT step.next_id, 'entity'::text, b.depth + 1, b.visited || step.next_id
      FROM bfs b,
           LATERAL (
             SELECT next_id FROM (
               SELECT (CASE WHEN ee.source_id = b.id THEN ee.target_id ELSE ee.source_id END) AS next_id
               FROM entity_edges ee
               WHERE b.kind = 'entity'
                 AND (ee.source_id = b.id OR ee.target_id = b.id)
                 AND ee.graph_id = ANY(${graphIds}::uuid[])
               UNION ALL
               SELECT me.entity_id AS next_id
               FROM episodic_edges me
               WHERE b.kind = 'episodic'
                 AND me.episodic_id = b.id
                 AND me.graph_id = ANY(${graphIds}::uuid[])
             ) candidates
             WHERE NOT (next_id = ANY(b.visited))
           ) step
      WHERE b.depth < ${depth}
    )
  `;
}
