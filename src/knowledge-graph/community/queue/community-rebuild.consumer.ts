import { InjectQueue, Processor } from '@nestjs/bullmq';
import { Job, Queue } from 'bullmq';

import { Span } from '@/observability';
import { BaseQueueConsumer } from '@/providers/queue/bullmq/base-queue-consumer';

import { CommunityService } from '../community.service';
import {
  COMMUNITY_REBUILD_QUEUE,
  type CommunityRebuildJobData,
} from './community-rebuild.queue';

/**
 * Runs the full Louvain rebuild for a graph. Concurrency is 1: a rebuild loads
 * the whole graph into RAM, and per-graph debounce collapses bursts.
 * Serializing keeps peak memory to a single graph.
 *
 * TODO(prod): rethink for AWS ECS. Likely split into one large-RAM task that
 * owns this queue (concurrency tunable to its RAM budget) and small replicas
 * that exclude it. If the rebuild role ever scales out, per-worker
 * concurrency=1 won't bound total RAM (different graphs have independent
 * debounce keys, so N workers = N graphs in memory) - need
 * queue.setGlobalConcurrency to cap cluster-wide in-flight rebuilds.
 */
@Processor(COMMUNITY_REBUILD_QUEUE, { concurrency: 1 })
export class CommunityRebuildConsumer extends BaseQueueConsumer {
  constructor(
    @InjectQueue(COMMUNITY_REBUILD_QUEUE) queue: Queue<CommunityRebuildJobData>,
    private readonly communityService: CommunityService,
  ) {
    super(CommunityRebuildConsumer.name, queue);
  }

  @Span('community-rebuild', { asLangfuseTrace: true })
  async process(
    job: Job<CommunityRebuildJobData>,
    _token?: string,
    signal?: AbortSignal,
  ): Promise<void> {
    const { userId, graphId } = job.data;
    try {
      await this.communityService.buildCommunities(
        userId,
        graphId,
        {
          userId,
          tags: ['knowledge-graph', 'community-rebuild', `graph:${graphId}`],
          metadata: { trigger: 'episode-debounced', jobId: job.id ?? '' },
        },
        signal,
      );
    } catch (err) {
      if (this.isTestTeardownError(err)) return;
      this.logger.error(`Community rebuild failed for graph ${graphId}`, err);
      throw err;
    }
  }
}
