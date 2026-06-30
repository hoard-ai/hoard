import { Injectable } from '@nestjs/common';

import { Prisma } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';
import type { Community } from '@/knowledge-graph/models';
import { FTS_NORM_NONE } from '@/knowledge-graph/search/types';
import {
  NodeName,
  type SearchBySimilarityParams,
  type SearchByTextParams,
} from '@/knowledge-graph/types';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import {
  chunkForBindParams,
  dedupeById,
  fromPgVector,
  toPgVector,
} from '../postgres-utils';
import { websearchTsquery } from '../sql-filter-builders';

type RawRow = {
  id: string;
  graph_id: string;
  name: string;
  summary: string | null;
  name_embedding: string | null;
  member_ids: string[] | null;
  created_at: Date;
  updated_at: Date;
};

const ROW_COLUMNS = Prisma.sql`id, graph_id, name, summary,
       name_embedding::text AS name_embedding,
       member_ids, created_at, updated_at`;

export type ClusterMatch =
  | { kind: 'matched-set'; communityId: Uuid; storedHashes: Record<string, string> }
  | { kind: 'unmatched' };

export type ExistingCommunitySnapshot = {
  id: Uuid;
  name: NodeName;
  memberIds: Uuid[];
  summary: string;
  nameEmbedding: number[] | null;
};

export type ClusterMatchResult = {
  matchesByClusterIndex: Map<number, ClusterMatch>;
  existing: ExistingCommunitySnapshot[];
};

@Injectable()
export class CommunityRepository {
  constructor(private readonly prisma: PrismaService) {}

  /**
   * member_signature is filled by the communities_set_member_signature trigger
   * (BEFORE INSERT OR UPDATE OF member_ids). Application code never sets it.
   */
  @Span()
  async save(c: Community): Promise<string> {
    await this.prisma.$executeRaw`
      INSERT INTO communities (
        id, graph_id, name, summary, name_embedding,
        member_ids, created_at, updated_at
      )
      VALUES (
        ${c.id}::uuid,
        ${c.graphId}::uuid,
        ${c.name},
        ${c.summary},
        ${toPgVector(c.nameEmbedding)}::vector,
        ${c.memberIds}::uuid[],
        ${c.createdAt},
        ${c.updatedAt}
      )
      ON CONFLICT (id) DO UPDATE SET
        name           = EXCLUDED.name,
        summary        = EXCLUDED.summary,
        name_embedding = EXCLUDED.name_embedding,
        member_ids     = EXCLUDED.member_ids,
        updated_at     = EXCLUDED.updated_at
    `;
    return c.id;
  }

  @Span()
  async saveBulk(cs: Community[]): Promise<void> {
    if (cs.length === 0) return;
    for (const chunk of chunkForBindParams(dedupeById(cs), 8)) {
      const rows = chunk.map(
        (c) => Prisma.sql`(
          ${c.id}::uuid,
          ${c.graphId}::uuid,
          ${c.name},
          ${c.summary},
          ${toPgVector(c.nameEmbedding)}::vector,
          ${c.memberIds}::uuid[],
          ${c.createdAt},
          ${c.updatedAt}
        )`,
      );
      await this.prisma.$executeRaw`
        INSERT INTO communities (
          id, graph_id, name, summary, name_embedding, member_ids, created_at, updated_at
        )
        VALUES ${Prisma.join(rows)}
        ON CONFLICT (id) DO UPDATE SET
          name           = EXCLUDED.name,
          summary        = EXCLUDED.summary,
          name_embedding = EXCLUDED.name_embedding,
          member_ids     = EXCLUDED.member_ids,
          updated_at     = EXCLUDED.updated_at
      `;
    }
  }

  /**
   * Atomic incremental update for both the rebuild incremental paths
   * (buildCommunities -> matched-drift / matched-superset) and the per-entity
   * update path. Replaces member_ids with the final array and refreshes
   * summary/name/embedding in one row-locked statement. Concurrent touches of
   * the same row serialize on the row, not on the worker, so no global cap is
   * needed.
   *
   * Both signature columns are recomputed by the BEFORE UPDATE trigger because
   * member_ids appears in the SET list - no app-side recompute needed.
   */
  @Span()
  async applyIncrementalUpdate(args: {
    id: Uuid;
    memberIds: Uuid[];
    name: NodeName;
    summary: string;
    nameEmbedding: number[] | null;
  }): Promise<void> {
    await this.prisma.$executeRaw`
      UPDATE communities
      SET member_ids     = ${args.memberIds}::uuid[],
          name           = ${args.name},
          summary        = ${args.summary},
          name_embedding = ${toPgVector(args.nameEmbedding)}::vector,
          updated_at     = now()
      WHERE id = ${args.id}::uuid
    `;
  }

  @Span()
  async deleteByIds(ids: Uuid[]): Promise<void> {
    if (ids.length === 0) return;
    await this.prisma.$executeRaw`
      DELETE FROM communities WHERE id = ANY(${ids}::uuid[])
    `;
  }

  @Span()
  async deleteByGraphId(graphId: Uuid): Promise<void> {
    await this.prisma.$executeRaw`
      DELETE FROM communities WHERE graph_id = ${graphId}::uuid
    `;
  }

  /**
   * Single round-trip: for each detected cluster, look up an existing community
   * with the same member_set_signature. Returns per-cluster match + the stored
   * member_summary_hashes snapshot (so the service can find drifted members in
   * TS by hashing fresh summaries and comparing). Also returns full snapshots
   * of every existing community in the graph so the service can do superset
   * matching for the addition-only case without a second round-trip.
   */
  @Span()
  async matchClusters(graphId: Uuid, clusters: Uuid[][]): Promise<ClusterMatchResult> {
    const existingRows = await this.prisma.$queryRaw<
      {
        id: string;
        name: string;
        member_ids: string[] | null;
        summary: string | null;
        name_embedding: string | null;
      }[]
    >`
      SELECT id, name, member_ids, summary, name_embedding::text AS name_embedding
      FROM communities
      WHERE graph_id = ${graphId}::uuid
    `;
    const existing: ExistingCommunitySnapshot[] = existingRows.map((r) => ({
      id: r.id as Uuid,
      name: r.name as NodeName,
      memberIds: (r.member_ids ?? []) as Uuid[],
      summary: r.summary ?? '',
      nameEmbedding: fromPgVector(r.name_embedding),
    }));

    if (clusters.length === 0) {
      return { matchesByClusterIndex: new Map(), existing };
    }

    const valuesRows = clusters.map(
      (memberIds, idx) => Prisma.sql`(${idx}::int, ${memberIds}::uuid[])`,
    );

    const rows = await this.prisma.$queryRaw<
      {
        idx: number;
        matched_id: string | null;
        stored_hashes: Record<string, string> | null;
      }[]
    >`
      WITH detected(idx, member_ids) AS (
        VALUES ${Prisma.join(valuesRows)}
      )
      SELECT d.idx,
             c.id::text AS matched_id,
             c.member_summary_hashes AS stored_hashes
      FROM detected d
      LEFT JOIN communities c
        ON c.graph_id = ${graphId}::uuid
       AND c.member_set_signature = compute_community_set_signature(d.member_ids)
      ORDER BY d.idx
    `;

    const matchesByClusterIndex = new Map<number, ClusterMatch>();
    for (const r of rows) {
      if (r.matched_id !== null && r.stored_hashes !== null) {
        matchesByClusterIndex.set(r.idx, {
          kind: 'matched-set',
          communityId: r.matched_id as Uuid,
          storedHashes: r.stored_hashes,
        });
      } else {
        matchesByClusterIndex.set(r.idx, { kind: 'unmatched' });
      }
    }
    return { matchesByClusterIndex, existing };
  }

  @Span()
  async searchByName(params: SearchByTextParams): Promise<Community[]> {
    const { query, graphIds, limit } = params;
    if (graphIds.length === 0) return [];
    const tsquery = websearchTsquery(query);

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT c.id, c.graph_id, c.name, c.summary,
             c.name_embedding::text AS name_embedding,
             c.member_ids, c.created_at, c.updated_at,
             ts_rank_cd(c.fts_vector, ${tsquery}, ${FTS_NORM_NONE}) AS score
      FROM communities c
      WHERE c.graph_id = ANY(${graphIds}::uuid[])
        AND c.fts_vector @@ ${tsquery}
      ORDER BY score DESC
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async searchBySimilarity(params: SearchBySimilarityParams): Promise<Community[]> {
    const { embedding, graphIds, limit, minScore } = params;
    if (graphIds.length === 0) return [];
    const vec = Prisma.sql`${toPgVector(embedding)}::vector`;

    // See EntityNodeRepository.searchBySimilarity for the bucket-filter rationale.
    const bucketArray = Prisma.sql`ARRAY[${Prisma.join(
      graphIds.map((g) => Prisma.sql`graph_diskann_bucket_for(${g}::uuid)`),
    )}]::smallint[]`;

    const rows = await this.prisma.$queryRaw<(RawRow & { score: number })[]>`
      SELECT c.id, c.graph_id, c.name, c.summary,
             c.name_embedding::text AS name_embedding,
             c.member_ids, c.created_at, c.updated_at,
             1 - (c.name_embedding <=> ${vec}) AS score
      FROM communities c
      WHERE c.graph_diskann_bucket && ${bucketArray}
        AND c.graph_id = ANY(${graphIds}::uuid[])
        AND c.name_embedding IS NOT NULL
        AND 1 - (c.name_embedding <=> ${vec}) >= ${minScore}
      ORDER BY c.name_embedding <=> ${vec}
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  /**
   * Incremental community-update lookup: which (if any) community already
   * contains this entity? Used to short-circuit when an entity is already
   * communified.
   */
  @Span()
  async findByMemberId(graphId: Uuid, entityId: Uuid): Promise<Community | null> {
    const rows = await this.prisma.$queryRaw<RawRow[]>`
      SELECT ${ROW_COLUMNS}
      FROM communities
      WHERE graph_id = ${graphId}::uuid AND ${entityId}::uuid = ANY(member_ids)
      LIMIT 1
    `;
    return rows.length > 0 ? this.mapRow(rows[0]) : null;
  }

  /**
   * Incremental community-update lookup: which communities contain any of the
   * given entity ids? Used to find the candidate community for a new entity by
   * looking at its neighbors' memberships.
   */
  @Span()
  async findByAnyMember(graphId: Uuid, entityIds: Uuid[]): Promise<Community[]> {
    if (entityIds.length === 0) return [];
    const rows = await this.prisma.$queryRaw<RawRow[]>`
      SELECT ${ROW_COLUMNS}
      FROM communities
      WHERE graph_id = ${graphId}::uuid AND member_ids && ${entityIds}::uuid[]
    `;
    return rows.map((r) => this.mapRow(r));
  }

  /**
   * Per-entity update path: fetch all community names in the graph so the
   * namer can pick a label that avoids collisions on first pass. Returned in
   * row order; deduping/exclusion is left to the caller.
   */
  @Span()
  async findNamesByGraphId(graphId: Uuid): Promise<NodeName[]> {
    const rows = await this.prisma.$queryRaw<{ name: string }[]>`
      SELECT name FROM communities WHERE graph_id = ${graphId}::uuid
    `;
    return rows.map((r) => r.name as NodeName);
  }

  private mapRow(row: RawRow): Community {
    return {
      id: row.id as Uuid,
      graphId: row.graph_id as Uuid,
      name: row.name as NodeName,
      summary: row.summary ?? '',
      nameEmbedding: fromPgVector(row.name_embedding),
      memberIds: (row.member_ids ?? []) as Uuid[],
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }
}
