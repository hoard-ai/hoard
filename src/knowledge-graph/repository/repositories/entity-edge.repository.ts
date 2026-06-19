import { Injectable } from '@nestjs/common';

import { Prisma } from '@generated/prisma/client';

import { Uuid } from '@/common/schemas';
import { EntityEdge } from '@/knowledge-graph/models';
import {
  FTS_NORM_NONE,
  MAX_SEARCH_DEPTH,
  SearchFilters,
} from '@/knowledge-graph/search/types';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import {
  RelationshipType,
  SearchByBfsParams,
  SearchBySimilarityParams,
  SearchByTextParams,
} from '../../types';
import { buildBfsCte } from '../bfs-cte';
import { fromPgVector, toPgVector } from '../pgvector-utils';
import { buildEdgeFilterClause, websearchTsquery } from '../sql-filter-builders';

type RawRow = {
  id: string;
  graph_id: string;
  source_id: string;
  target_id: string;
  name: string;
  fact: string;
  fact_embedding: string | null;
  attributes: unknown;
  episodes: string[] | null;
  valid_at: Date | null;
  invalid_at: Date | null;
  expired_at: Date | null;
  created_at: Date;
};

@Injectable()
export class EntityEdgeRepository {
  constructor(private readonly prisma: PrismaService) {}

  @Span()
  async save(edge: EntityEdge, tx?: Prisma.TransactionClient): Promise<string> {
    const db = tx ?? this.prisma;
    await db.$executeRaw`
      INSERT INTO entity_edges (
        id, graph_id, source_id, target_id, name, fact,
        fact_embedding, attributes, episodes, valid_at, invalid_at, expired_at, created_at
      ) VALUES (
        ${edge.id}::uuid,
        ${edge.graphId}::uuid,
        ${edge.sourceNodeId}::uuid,
        ${edge.targetNodeId}::uuid,
        ${edge.name},
        ${edge.fact},
        ${toPgVector(edge.factEmbedding)}::vector,
        ${JSON.stringify(edge.attributes)}::jsonb,
        ${edge.episodes}::uuid[],
        ${edge.validAt},
        ${edge.invalidAt},
        ${edge.expiredAt},
        ${edge.createdAt}
      )
      ON CONFLICT (id) DO UPDATE SET
        graph_id       = EXCLUDED.graph_id,
        source_id      = EXCLUDED.source_id,
        target_id      = EXCLUDED.target_id,
        name           = EXCLUDED.name,
        fact           = EXCLUDED.fact,
        fact_embedding = EXCLUDED.fact_embedding,
        attributes     = EXCLUDED.attributes,
        episodes       = EXCLUDED.episodes,
        valid_at       = EXCLUDED.valid_at,
        invalid_at     = EXCLUDED.invalid_at,
        expired_at     = EXCLUDED.expired_at
    `;
    return edge.id;
  }

  @Span()
  async saveBulk(edges: EntityEdge[], tx?: Prisma.TransactionClient): Promise<void> {
    if (edges.length === 0) return;
    for (const edge of edges) {
      await this.save(edge, tx);
    }
  }

  @Span()
  async deleteByIds(ids: Uuid[]): Promise<void> {
    if (ids.length === 0) return;
    await this.prisma.entityEdge.deleteMany({ where: { id: { in: ids } } });
  }

  /**
   * Undirected, aggregated neighbor counts for label propagation. Postgres
   * collapses parallel edges into edge_count and normalizes direction so each
   * unordered pair appears once.
   */
  @Span()
  async findAggregatedNeighborCounts(
    graphId: Uuid,
  ): Promise<Array<{ a: Uuid; b: Uuid; edgeCount: number }>> {
    const rows = await this.prisma.$queryRaw<
      { a: string; b: string; edge_count: number }[]
    >`
      SELECT LEAST(source_id, target_id)::uuid    AS a,
             GREATEST(source_id, target_id)::uuid AS b,
             count(*)::int                        AS edge_count
      FROM entity_edges
      WHERE graph_id = ${graphId}::uuid AND source_id <> target_id
      GROUP BY a, b
    `;
    return rows.map((r) => ({
      a: r.a as Uuid,
      b: r.b as Uuid,
      edgeCount: r.edge_count,
    }));
  }

  /**
   * Raw edge count for a graph. Used by community maintenance routing as a
   * cheap memory proxy (the rebuild loads aggregated edges into RAM).
   */
  @Span()
  async countForGraph(graphId: Uuid): Promise<number> {
    const rows = await this.prisma.$queryRaw<{ count: number }[]>`
      SELECT count(*)::int AS count FROM entity_edges WHERE graph_id = ${graphId}::uuid
    `;
    return rows[0]?.count ?? 0;
  }

  /**
   * Undirected neighbor ids for a single entity. Used by the incremental
   * community update path to find which community a new entity should join.
   */
  @Span()
  async findNeighborIds(entityId: Uuid): Promise<Uuid[]> {
    const rows = await this.prisma.$queryRaw<{ neighbor_id: string }[]>`
      SELECT DISTINCT CASE
               WHEN source_id = ${entityId}::uuid THEN target_id
               ELSE source_id
             END AS neighbor_id
      FROM entity_edges
      WHERE (source_id = ${entityId}::uuid OR target_id = ${entityId}::uuid)
        AND source_id <> target_id
    `;
    return rows.map((r) => r.neighbor_id as Uuid);
  }

  @Span()
  async getIdsForEpisodeDeletion(episodeId: Uuid): Promise<Uuid[]> {
    // Only edges whose FIRST episode in the
    // ordered `episodes` array matches are considered. Edges that merely
    // accumulated this episode as a later contributor are intentionally kept.
    const rows = await this.prisma.$queryRaw<{ id: Uuid }[]>`
      SELECT id
      FROM entity_edges
      WHERE episodes[1] = ${episodeId}::uuid
    `;
    return rows.map((r) => r.id);
  }

  @Span()
  async getBetweenNodes(sourceId: Uuid, targetId: Uuid): Promise<EntityEdge[]> {
    const rows = await this.prisma.$queryRaw<RawRow[]>`
      SELECT id, graph_id, source_id, target_id, name, fact,
             fact_embedding::text AS fact_embedding, attributes, episodes,
             valid_at, invalid_at, expired_at, created_at
      FROM entity_edges
      WHERE source_id = ${sourceId}::uuid AND target_id = ${targetId}::uuid
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchByFact(params: SearchByTextParams): Promise<EntityEdge[]> {
    const { query, graphIds, limit } = params;
    if (graphIds.length === 0) return [];
    const tsquery = websearchTsquery(query);

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT e.id, e.graph_id, e.source_id, e.target_id, e.name, e.fact,
             e.fact_embedding::text AS fact_embedding, e.attributes, e.episodes,
             e.valid_at, e.invalid_at, e.expired_at, e.created_at,
             ts_rank_cd(e.fts_vector, ${tsquery}, ${FTS_NORM_NONE}) AS score
      FROM entity_edges e
      WHERE e.graph_id = ANY(${graphIds}::uuid[])
        AND e.fts_vector @@ ${tsquery}
      ORDER BY score DESC
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchBySimilarity(
    params: SearchBySimilarityParams,
    filters?: SearchFilters,
  ): Promise<EntityEdge[]> {
    const { embedding, graphIds, limit, minScore } = params;
    if (graphIds.length === 0) return [];
    const vec = Prisma.sql`${toPgVector(embedding)}::vector`;
    const filterSql = buildEdgeFilterClause(filters, 'e');

    // See comment in ./entity-node.repository.ts
    const bucketArray = Prisma.sql`ARRAY[${Prisma.join(
      graphIds.map((g) => Prisma.sql`graph_diskann_bucket_for(${g}::uuid)`),
    )}]::smallint[]`;

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT e.id, e.graph_id, e.source_id, e.target_id, e.name, e.fact,
             e.fact_embedding::text AS fact_embedding, e.attributes, e.episodes,
             e.valid_at, e.invalid_at, e.expired_at, e.created_at,
             1 - (e.fact_embedding <=> ${vec}) AS score
      FROM entity_edges e
      WHERE e.graph_diskann_bucket && ${bucketArray}
        AND e.graph_id = ANY(${graphIds}::uuid[])
        AND e.fact_embedding IS NOT NULL
        AND 1 - (e.fact_embedding <=> ${vec}) >= ${minScore}
        ${filterSql}
      ORDER BY e.fact_embedding <=> ${vec}
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchByBfs(
    params: SearchByBfsParams,
    filters?: SearchFilters,
  ): Promise<EntityEdge[]> {
    const { originNodeIds, graphIds, limit, maxDepth } = params;
    if (originNodeIds.length === 0) return [];
    const depth = maxDepth ?? MAX_SEARCH_DEPTH;
    const filterSql = buildEdgeFilterClause(filters, 'e');

    const cte = buildBfsCte(originNodeIds, graphIds, depth);
    const rows = await this.prisma.$queryRaw<RawRow[]>`
      ${cte}
      SELECT DISTINCT e.id, e.graph_id, e.source_id, e.target_id, e.name, e.fact,
                      e.fact_embedding::text AS fact_embedding, e.attributes, e.episodes,
                      e.valid_at, e.invalid_at, e.expired_at, e.created_at
      FROM entity_edges e
      WHERE e.graph_id = ANY(${graphIds}::uuid[])
        AND e.source_id IN (SELECT id FROM bfs)
        AND e.target_id IN (SELECT id FROM bfs)
        ${filterSql}
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  private mapRow(row: RawRow): EntityEdge {
    return {
      id: row.id as Uuid,
      graphId: row.graph_id as Uuid,
      sourceNodeId: row.source_id as Uuid,
      targetNodeId: row.target_id as Uuid,
      name: row.name as RelationshipType,
      fact: row.fact,
      factEmbedding: fromPgVector(row.fact_embedding),
      episodes: (row.episodes ?? []) as Uuid[],
      validAt: row.valid_at,
      invalidAt: row.invalid_at,
      expiredAt: row.expired_at,
      createdAt: row.created_at,
      attributes:
        row.attributes && typeof row.attributes === 'object'
          ? (row.attributes as Record<string, unknown>)
          : {},
    };
  }
}
