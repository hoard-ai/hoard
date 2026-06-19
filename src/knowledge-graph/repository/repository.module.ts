import { Module } from '@nestjs/common';

import {
  CommunityRepository,
  EntityEdgeRepository,
  EntityNodeRepository,
  EpisodicEdgeRepository,
  EpisodicNodeRepository,
  SagaRepository,
} from './repositories';

const repositories = [
  CommunityRepository,
  EntityNodeRepository,
  EntityEdgeRepository,
  EpisodicNodeRepository,
  EpisodicEdgeRepository,
  SagaRepository,
];

// PrismaModule is @Global() - no explicit import needed.
@Module({
  providers: repositories,
  exports: repositories,
})
export class RepositoryModule {}
