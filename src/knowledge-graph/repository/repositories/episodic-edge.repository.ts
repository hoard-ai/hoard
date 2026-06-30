import { Injectable } from '@nestjs/common';

import { Prisma } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';
import type { EpisodicEdge } from '@/knowledge-graph/models';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import { chunkForBindParams, dedupeById } from '../postgres-utils';

@Injectable()
export class EpisodicEdgeRepository {
  constructor(private readonly prisma: PrismaService) {}

  @Span()
  async save(edge: EpisodicEdge, tx?: Prisma.TransactionClient): Promise<string> {
    const db = tx ?? this.prisma;
    await db.episodicEdge.upsert({
      where: { id: edge.id },
      create: {
        id: edge.id,
        graphId: edge.graphId,
        episodicId: edge.sourceNodeId,
        entityId: edge.targetNodeId,
        createdAt: edge.createdAt,
      },
      update: {
        graphId: edge.graphId,
        episodicId: edge.sourceNodeId,
        entityId: edge.targetNodeId,
      },
    });
    return edge.id;
  }

  @Span()
  async saveBulk(edges: EpisodicEdge[], tx?: Prisma.TransactionClient): Promise<void> {
    if (edges.length === 0) return;
    const db = tx ?? this.prisma;
    for (const chunk of chunkForBindParams(dedupeById(edges), 5)) {
      const rows = chunk.map(
        (edge) => Prisma.sql`(
          ${edge.id}::uuid,
          ${edge.graphId}::uuid,
          ${edge.sourceNodeId}::uuid,
          ${edge.targetNodeId}::uuid,
          ${edge.createdAt}
        )`,
      );
      await db.$executeRaw`
        INSERT INTO episodic_edges (id, graph_id, episodic_id, entity_id, created_at)
        VALUES ${Prisma.join(rows)}
        ON CONFLICT (id) DO UPDATE SET
          graph_id    = EXCLUDED.graph_id,
          episodic_id = EXCLUDED.episodic_id,
          entity_id   = EXCLUDED.entity_id
      `;
    }
  }

  @Span()
  async deleteBySourceId(episodeId: Uuid): Promise<void> {
    await this.prisma.episodicEdge.deleteMany({ where: { episodicId: episodeId } });
  }
}
