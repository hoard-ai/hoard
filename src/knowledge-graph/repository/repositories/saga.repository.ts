import { Injectable } from '@nestjs/common';

import { Prisma, Saga as PrismaSaga } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';
import type { Saga } from '@/knowledge-graph/models';
import { Span } from '@/observability';
import { PrismaService } from '@/providers/database/postgres/prisma.service';

import { NodeLabels, NodeName } from '../../types';

@Injectable()
export class SagaRepository {
  constructor(private readonly prisma: PrismaService) {}

  @Span()
  async save(saga: Saga): Promise<string> {
    await this.prisma.saga.upsert({
      where: { id: saga.id },
      create: {
        id: saga.id,
        graphId: saga.graphId,
        name: saga.name,
        labels: saga.labels,
        summary: saga.summary,
        lastSummarizedAt: saga.lastSummarizedAt,
        createdAt: saga.createdAt,
      },
      update: {
        graphId: saga.graphId,
        name: saga.name,
        labels: saga.labels,
        summary: saga.summary,
        lastSummarizedAt: saga.lastSummarizedAt,
      },
    });
    return saga.id;
  }

  @Span()
  async createIfNotExists(saga: Saga, tx?: Prisma.TransactionClient): Promise<void> {
    const db = tx ?? this.prisma;
    await db.saga.createMany({
      data: {
        id: saga.id,
        graphId: saga.graphId,
        name: saga.name,
        labels: saga.labels,
        summary: saga.summary,
        lastSummarizedAt: saga.lastSummarizedAt,
        createdAt: saga.createdAt,
      },
      skipDuplicates: true,
    });
  }

  @Span()
  async getById(id: Uuid): Promise<Saga | null> {
    const row = await this.prisma.saga.findUnique({ where: { id: id } });
    return row ? this.mapRow(row) : null;
  }

  private mapRow(row: PrismaSaga): Saga {
    return {
      id: row.id as Uuid,
      graphId: row.graphId as Uuid,
      name: row.name as NodeName,
      labels: (row.labels ?? []) as NodeLabels,
      summary: row.summary ?? '',
      lastSummarizedAt: row.lastSummarizedAt,
      createdAt: row.createdAt,
    };
  }
}
