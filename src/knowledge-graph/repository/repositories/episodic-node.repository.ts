import { Injectable } from '@nestjs/common';

import { Prisma, EpisodicNode as PrismaEpisodicNode } from '@generated/prisma/client';

import { Uuid } from '@/common/schemas';
import { EpisodicNode } from '@/knowledge-graph/models';
import { FTS_NORM_LOG_LENGTH } from '@/knowledge-graph/search/types';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import {
  EpisodeType,
  NodeLabels,
  NodeName,
  RetrieveEpisodesParams,
  SearchByTextParams,
} from '../../types';
import { websearchTsquery } from '../sql-filter-builders';

type Row = Pick<
  PrismaEpisodicNode,
  | 'id'
  | 'graphId'
  | 'name'
  | 'labels'
  | 'source'
  | 'sourceDescription'
  | 'content'
  | 'validAt'
  | 'sagaId'
  | 'createdAt'
>;

@Injectable()
export class EpisodicNodeRepository {
  constructor(private readonly prisma: PrismaService) {}

  @Span()
  async save(node: EpisodicNode, tx?: Prisma.TransactionClient): Promise<string> {
    const db = tx ?? this.prisma;
    await db.episodicNode.upsert({
      where: { id: node.id },
      create: {
        id: node.id,
        graphId: node.graphId,
        name: node.name,
        labels: node.labels,
        source: node.source,
        sourceDescription: node.sourceDescription,
        content: node.content,
        validAt: node.validAt,
        sagaId: node.sagaId,
        createdAt: node.createdAt,
      },
      update: {
        graphId: node.graphId,
        name: node.name,
        labels: node.labels,
        source: node.source,
        sourceDescription: node.sourceDescription,
        content: node.content,
        validAt: node.validAt,
        sagaId: node.sagaId,
      },
    });
    return node.id;
  }

  @Span()
  async saveBulk(nodes: EpisodicNode[], tx?: Prisma.TransactionClient): Promise<void> {
    if (nodes.length === 0) return;
    for (const node of nodes) await this.save(node, tx);
  }

  @Span()
  async delete(id: Uuid): Promise<void> {
    await this.prisma.episodicNode.delete({ where: { id: id } });
  }

  @Span()
  async getById(id: Uuid): Promise<EpisodicNode | null> {
    const row = await this.prisma.episodicNode.findUnique({ where: { id: id } });
    return row ? this.mapRow(row) : null;
  }

  @Span()
  async retrieveEpisodes(params: RetrieveEpisodesParams): Promise<EpisodicNode[]> {
    const { referenceTime, graphIds, source, sagaId, lastN } = params;
    const rows = await this.prisma.episodicNode.findMany({
      where: {
        validAt: { lte: referenceTime },
        graphId: { in: graphIds },
        ...(source ? { source } : {}),
        ...(sagaId ? { sagaId } : {}),
      },
      orderBy: [{ validAt: 'desc' }, { createdAt: 'desc' }],
      take: lastN,
    });
    return rows.map((r) => this.mapRow(r));
  }

  @Span()
  async getMentionedEntityIds(episodeId: Uuid): Promise<Uuid[]> {
    const rows = await this.prisma.episodicEdge.findMany({
      where: { episodicId: episodeId },
      select: { entityId: true },
    });
    return rows.map((r) => r.entityId as Uuid);
  }

  @Span()
  async searchByContent(params: SearchByTextParams): Promise<EpisodicNode[]> {
    const { query, graphIds, limit } = params;
    if (graphIds.length === 0) return [];
    const tsquery = websearchTsquery(query);
    const rows = await this.prisma.$queryRaw<(Row & { score: number })[]>`
      SELECT en.id,
             en.graph_id           AS "graphId",
             en.name,
             en.labels,
             en.source,
             en.source_description AS "sourceDescription",
             en.content,
             en.valid_at           AS "validAt",
             en.saga_id            AS "sagaId",
             en.created_at         AS "createdAt",
             ts_rank_cd(en.fts_vector, ${tsquery}, ${FTS_NORM_LOG_LENGTH}) AS score
      FROM episodic_nodes en
      WHERE en.graph_id = ANY(${graphIds}::uuid[])
        AND en.fts_vector @@ ${tsquery}
      ORDER BY score DESC
      LIMIT ${limit}
    `;
    return rows.map((r) => this.mapRow(r));
  }

  // TODO: Remove need for this with chunk search
  /**
   * ts_headline excerpts for the given episodes, highlighted against the query.
   * Computed only for already-selected top-K episodes (ts_headline reads the
   * full document and is slow). Neutral [[…]] delimiters (no HTML).
   */
  @Span()
  async searchSnippets(ids: Uuid[], query: string): Promise<Map<Uuid, string>> {
    if (ids.length === 0) return new Map();
    const tsquery = websearchTsquery(query);
    const rows = await this.prisma.$queryRaw<{ id: string; snippet: string }[]>`
      SELECT en.id,
             ts_headline('english', en.content, ${tsquery},
               'StartSel=[[, StopSel=]], MaxFragments=2, MinWords=8, MaxWords=30') AS snippet
      FROM episodic_nodes en
      WHERE en.id = ANY(${ids}::uuid[])
    `;
    return new Map(rows.map((r) => [r.id as Uuid, r.snippet]));
  }

  private mapRow(row: Row): EpisodicNode {
    return {
      id: row.id as Uuid,
      name: row.name as NodeName,
      graphId: row.graphId as Uuid,
      labels: (row.labels ?? []) as NodeLabels,
      createdAt: row.createdAt,
      source: (row.source as EpisodeType) ?? EpisodeType.text,
      sourceDescription: row.sourceDescription,
      content: row.content ?? '',
      validAt: row.validAt,
      sagaId: (row.sagaId as Uuid | null) ?? null,
    };
  }
}
