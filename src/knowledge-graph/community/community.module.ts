import { BullModule } from '@nestjs/bullmq';
import { Module } from '@nestjs/common';

import { CommunityConfigModule } from '@/config/community';
import { KnowledgeGraphConfigModule } from '@/config/knowledge-graph';
import { LlmModule } from '@/llm/llm.module';

import { EmbeddingModule } from '../embedding/embedding.module';
import { RepositoryModule } from '../repository/repository.module';
import { CommunityMaintenanceService } from './community-maintenance.service';
import { CommunityService } from './community.service';
import {
  COMMUNITY_REBUILD_QUEUE,
  COMMUNITY_UPDATE_QUEUE,
  CommunityRebuildConsumer,
  CommunityRebuildQueueService,
  CommunityUpdateConsumer,
  CommunityUpdateQueueService,
} from './queue';

@Module({
  imports: [
    BullModule.registerQueue(
      { name: COMMUNITY_UPDATE_QUEUE },
      { name: COMMUNITY_REBUILD_QUEUE },
    ),
    CommunityConfigModule,
    KnowledgeGraphConfigModule,
    RepositoryModule,
    LlmModule,
    EmbeddingModule,
  ],
  providers: [
    CommunityService,
    CommunityMaintenanceService,
    CommunityUpdateQueueService,
    CommunityUpdateConsumer,
    CommunityRebuildQueueService,
    CommunityRebuildConsumer,
  ],
  exports: [CommunityService, CommunityMaintenanceService],
})
export class CommunityModule {}
