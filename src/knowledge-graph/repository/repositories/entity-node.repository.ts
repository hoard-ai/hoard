import { Injectable } from '@nestjs/common';

import { Prisma } from '@generated/prisma/client';

import { Uuid } from '@/common/schemas';
import { EntityNode } from '@/knowledge-graph/models';
import {
  FTS_NORM_NONE,
  MAX_SEARCH_DEPTH,
  SearchFilters,
} from '@/knowledge-graph/search/types';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import {
  NodeLabels,
  NodeName,
  SearchByBfsParams,
  SearchBySimilarityParams,
  SearchByTextParams,
} from '../../types';
import { buildBfsCte } from '../bfs-cte';
import { fromPgVector, toPgVector } from '../pgvector-utils';
import { buildNodeFilterClause, websearchTsquery } from '../sql-filter-builders';

type RawRow = {
  id: string;
  graph_id: string;
  name: string;
  created_at: Date;
  summary: string | null;
  attributes: unknown;
  name_embedding: string | null;
  labels: string[] | null;
};

@Injectable()
export class EntityNodeRepository {
  constructor(private readonly prisma: PrismaService) {}

  @Span()
  async save(node: EntityNode, tx?: Prisma.TransactionClient): Promise<string> {
    const db = tx ?? this.prisma;
    await db.$executeRaw`
      INSERT INTO entity_nodes (id, graph_id, name, summary, attributes, labels, name_embedding, created_at)
      VALUES (
        ${node.id}::uuid,
        ${node.graphId}::uuid,
        ${node.name},
        ${node.summary},
        ${JSON.stringify(node.attributes)}::jsonb,
        ${node.labels}::text[],
        ${toPgVector(node.nameEmbedding)}::vector,
        ${node.createdAt}
      )
      ON CONFLICT (id) DO UPDATE SET
        graph_id       = EXCLUDED.graph_id,
        name           = EXCLUDED.name,
        summary        = EXCLUDED.summary,
        attributes     = EXCLUDED.attributes,
        labels         = EXCLUDED.labels,
        name_embedding = EXCLUDED.name_embedding
    `;
    return node.id;
  }

  @Span()
  async saveBulk(nodes: EntityNode[], tx?: Prisma.TransactionClient): Promise<void> {
    if (nodes.length === 0) return;
    // Sequential single-row UPSERTs keep parameter binding simple and stay
    // within Postgres' bind-parameter limit on large batches. The save() above
    // is itself a single round trip; bulk amortizes only client-side overhead.
    for (const node of nodes) {
      await this.save(node, tx);
    }
  }

  /**
   * Structure-only load for the minimal graph used by community detection.
   * Skips name, summary, attributes, embeddings - returns just IDs sorted
   * ascending for deterministic graph construction.
   */
  @Span()
  async findIdsForGraph(graphId: Uuid): Promise<Uuid[]> {
    const rows = await this.prisma.$queryRaw<{ id: string }[]>`
      SELECT id FROM entity_nodes WHERE graph_id = ${graphId}::uuid ORDER BY id ASC
    `;
    return rows.map((r) => r.id as Uuid);
  }

  /**
   * Node count for a graph. Used by community maintenance routing to decide
   * between a full rebuild and the incremental fallback.
   */
  @Span()
  async countForGraph(graphId: Uuid): Promise<number> {
    const rows = await this.prisma.$queryRaw<{ count: number }[]>`
      SELECT count(*)::int AS count FROM entity_nodes WHERE graph_id = ${graphId}::uuid
    `;
    return rows[0]?.count ?? 0;
  }

  /**
   * Batch (id, summary) load for community summarization. Skips embeddings,
   * attributes, labels - the tournament prompt only needs the summary text.
   */
  @Span()
  async findSummariesByIds(ids: Uuid[]): Promise<Array<{ id: Uuid; summary: string }>> {
    if (ids.length === 0) return [];
    const rows = await this.prisma.$queryRaw<{ id: string; summary: string | null }[]>`
      SELECT id, summary FROM entity_nodes WHERE id = ANY(${ids}::uuid[])
    `;
    return rows.map((r) => ({ id: r.id as Uuid, summary: r.summary ?? '' }));
  }

  @Span()
  async deleteIfSoleMentioned(nodeId: Uuid): Promise<void> {
    await this.prisma.$executeRaw`
      DELETE FROM entity_nodes en
      WHERE en.id = ${nodeId}::uuid
        AND (SELECT COUNT(*) FROM episodic_edges ee WHERE ee.entity_id = en.id) = 1
    `;
  }

  @Span()
  async searchByName(
    params: SearchByTextParams,
    filters?: SearchFilters,
  ): Promise<EntityNode[]> {
    const { query, graphIds, limit } = params;
    if (graphIds.length === 0) return [];
    const tsquery = websearchTsquery(query);
    const filterSql = buildNodeFilterClause(filters, 'n');

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT n.id, n.graph_id, n.name, n.summary, n.attributes, n.labels,
             n.name_embedding::text AS name_embedding, n.created_at,
             ts_rank_cd(n.fts_vector, ${tsquery}, ${FTS_NORM_NONE}) AS score
      FROM entity_nodes n
      WHERE n.graph_id = ANY(${graphIds}::uuid[])
        AND n.fts_vector @@ ${tsquery}
        ${filterSql}
      ORDER BY score DESC
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchBySimilarity(
    params: SearchBySimilarityParams,
    filters?: SearchFilters,
  ): Promise<EntityNode[]> {
    const { embedding, graphIds, limit, minScore } = params;
    if (graphIds.length === 0) return [];
    const vec = Prisma.sql`${toPgVector(embedding)}::vector`;
    const filterSql = buildNodeFilterClause(filters, 'n');

    // graph_diskann_bucket && ARRAY[graph_diskann_bucket_for(id1), graph_diskann_bucket_for(id2), ...]
    // narrows the diskann scan to label buckets containing any of our graphs.
    // graph_id = ANY(...) post-filters out bucket-mates from other graphs.
    // https://github.com/timescale/pgvectorscale?tab=readme-ov-file#label-based-filtering-with-diskann
    const bucketArray = Prisma.sql`ARRAY[${Prisma.join(
      graphIds.map((g) => Prisma.sql`graph_diskann_bucket_for(${g}::uuid)`),
    )}]::smallint[]`;

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT n.id, n.graph_id, n.name, n.summary, n.attributes, n.labels,
             n.name_embedding::text AS name_embedding, n.created_at,
             1 - (n.name_embedding <=> ${vec}) AS score
      FROM entity_nodes n
      WHERE n.graph_diskann_bucket && ${bucketArray}
        AND n.graph_id = ANY(${graphIds}::uuid[])
        AND n.name_embedding IS NOT NULL
        AND 1 - (n.name_embedding <=> ${vec}) >= ${minScore}
        ${filterSql}
      ORDER BY n.name_embedding <=> ${vec}
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchByBfs(
    params: SearchByBfsParams,
    filters?: SearchFilters,
  ): Promise<EntityNode[]> {
    const { originNodeIds, graphIds, limit, maxDepth } = params;
    if (originNodeIds.length === 0) return [];
    const depth = maxDepth ?? MAX_SEARCH_DEPTH;
    const filterSql = buildNodeFilterClause(filters, 'reachable');
    const cte = buildBfsCte(originNodeIds, graphIds, depth);
    const rows = await this.prisma.$queryRaw<RawRow[]>`
      ${cte}
      SELECT DISTINCT reachable.id, reachable.graph_id, reachable.name, reachable.summary,
                      reachable.attributes, reachable.labels,
                      reachable.name_embedding::text AS name_embedding, reachable.created_at
      FROM bfs b
      JOIN entity_nodes reachable ON reachable.id = b.id
      WHERE reachable.graph_id = ANY(${graphIds}::uuid[])
        AND b.depth > 0
        ${filterSql}
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  /**
   * Currently emits only 1-hop scores (score=1 for direct neighbors, absent otherwise).
   * The reranker treats score as graph distance with harmonic decay (1/d), so this can be
   * extended to multi-hop via BFS (e.g. buildBfsCte) without changing the caller contract.
   */
  @Span()
  async getNodeDistanceScores(
    nodeIds: Uuid[],
    centerNodeId: Uuid,
  ): Promise<{ id: Uuid; score: number }[]> {
    if (nodeIds.length === 0) return [];
    const rows = await this.prisma.$queryRaw<{ id: Uuid; score: number }[]>`
      SELECT DISTINCT n.id, 1 AS score
      FROM UNNEST(${nodeIds}::uuid[]) AS n(id)
      JOIN entity_edges ee
        ON (ee.source_id = ${centerNodeId}::uuid AND ee.target_id = n.id)
        OR (ee.target_id = ${centerNodeId}::uuid AND ee.source_id = n.id)
    `;
    return rows;
  }

  @Span()
  async getEpisodeMentionCounts(nodeIds: Uuid[]): Promise<{ id: Uuid; score: number }[]> {
    if (nodeIds.length === 0) return [];
    const rows = await this.prisma.$queryRaw<{ id: Uuid; score: number }[]>`
      SELECT n.id, COUNT(ee.id)::int AS score
      FROM UNNEST(${nodeIds}::uuid[]) AS n(id)
      LEFT JOIN episodic_edges ee ON ee.entity_id = n.id
      GROUP BY n.id
    `;
    return rows;
  }

  private mapRow(row: RawRow): EntityNode {
    const attrs = row.attributes;
    return {
      id: row.id as Uuid,
      name: row.name as NodeName,
      graphId: row.graph_id as Uuid,
      createdAt: row.created_at,
      summary: row.summary ?? '',
      attributes:
        attrs && typeof attrs === 'object' ? (attrs as Record<string, unknown>) : {},
      nameEmbedding: fromPgVector(row.name_embedding),
      labels: (row.labels ?? []) as NodeLabels,
    };
  }
}
