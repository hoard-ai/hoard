import { Injectable, Logger } from '@nestjs/common';

import type { Uuid } from '@/common/schemas';
import { CommunityConfigService } from '@/config/community';
import { metricsOnResult, Span, type SpanMetrics } from '@/observability';

import { EntityEdgeRepository, EntityNodeRepository } from '../repository/repositories';
import { CommunityRebuildQueueService, CommunityUpdateQueueService } from './queue';

/**
 * Decides how to maintain communities after episode ingestion: schedule a
 * debounced full Louvain rebuild while a graph is small enough, or fall back to
 * the incremental per-entity update path once it crosses the configured node
 * limit (off by default). Owns the size policy so CommunityService stays pure
 * compute and EpisodeService stays ignorant of queues and limits.
 */
@Injectable()
export class CommunityMaintenanceService {
  private readonly logger = new Logger(CommunityMaintenanceService.name);

  constructor(
    private readonly config: CommunityConfigService,
    private readonly entityNodeRepo: EntityNodeRepository,
    private readonly entityEdgeRepo: EntityEdgeRepository,
    private readonly rebuildQueue: CommunityRebuildQueueService,
    private readonly updateQueue: CommunityUpdateQueueService,
  ) {}

  async scheduleMaintenance(
    userId: Uuid,
    graphId: Uuid,
    entityIds: Uuid[],
  ): Promise<void> {
    await this.scheduleMaintenanceImpl(userId, graphId, entityIds);
  }

  @Span('scheduleCommunityMaintenance', { onResult: metricsOnResult })
  private async scheduleMaintenanceImpl(
    userId: Uuid,
    graphId: Uuid,
    entityIds: Uuid[],
  ): Promise<{ metrics: SpanMetrics }> {
    const baseMetrics: SpanMetrics = {
      'user.id': userId,
      'graph.id': graphId,
      'entities.touched': entityIds.length,
      'limit.enabled': this.config.rebuildLimitEnabled,
      'limit.maxNodes': this.config.rebuildMaxNodes,
    };
    if (entityIds.length === 0) {
      return {
        metrics: { ...baseMetrics, skipped: true, 'skipped.reason': 'no-entities' },
      };
    }
    const [nodeCount, edgeCount] = await Promise.all([
      this.entityNodeRepo.countForGraph(graphId),
      this.entityEdgeRepo.countForGraph(graphId),
    ]);

    const overLimit =
      this.config.rebuildLimitEnabled && nodeCount > this.config.rebuildMaxNodes;

    if (overLimit) {
      this.logger.warn(
        `Graph ${graphId} has ${nodeCount} nodes (limit ${this.config.rebuildMaxNodes}); ` +
          `using incremental community updates instead of a full rebuild.`,
      );
      await this.updateQueue.enqueue({ userId, graphId, entityIds });
    } else {
      await this.rebuildQueue.enqueue({ userId, graphId }, this.config.rebuildDebounceMs);
    }

    return {
      metrics: {
        ...baseMetrics,
        'nodes.count': nodeCount,
        'edges.count': edgeCount,
        path: overLimit ? 'incremental' : 'rebuild',
      },
    };
  }
}
