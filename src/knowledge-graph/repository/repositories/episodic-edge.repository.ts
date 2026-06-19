import { Injectable } from '@nestjs/common';

import { Prisma } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';
import type { EpisodicEdge } from '@/knowledge-graph/models';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

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
    for (const edge of edges) await this.save(edge, tx);
  }

  @Span()
  async deleteBySourceId(episodeId: Uuid): Promise<void> {
    await this.prisma.episodicEdge.deleteMany({ where: { episodicId: episodeId } });
  }
}
