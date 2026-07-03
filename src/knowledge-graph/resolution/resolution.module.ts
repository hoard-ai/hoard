import { Module } from '@nestjs/common';

import { KnowledgeGraphConfigModule } from '@/config/knowledge-graph';

import { RepositoryModule } from '../repository/repository.module';
import { EdgeResolutionService } from './edge-resolution.service';
import { NodeResolutionService } from './node-resolution.service';

@Module({
  imports: [RepositoryModule, KnowledgeGraphConfigModule],
  providers: [NodeResolutionService, EdgeResolutionService],
  exports: [NodeResolutionService, EdgeResolutionService],
})
export class ResolutionModule {}
