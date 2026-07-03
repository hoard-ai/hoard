import { Module } from '@nestjs/common';

import { KnowledgeGraphConfigModule } from '@/config/knowledge-graph';
import { LlmModule } from '@/llm/llm.module';

import { CommunityModule } from '../community/community.module';
import { EmbeddingModule } from '../embedding/embedding.module';
import { ExtractionModule } from '../extraction/extraction.module';
import { RepositoryModule } from '../repository/repository.module';
import { ResolutionModule } from '../resolution/resolution.module';
import { EpisodeService } from './episode.service';

@Module({
  imports: [
    LlmModule,
    EmbeddingModule,
    ExtractionModule,
    ResolutionModule,
    CommunityModule,
    RepositoryModule,
    KnowledgeGraphConfigModule,
  ],
  providers: [EpisodeService],
  exports: [EpisodeService],
})
export class EpisodeModule {}
