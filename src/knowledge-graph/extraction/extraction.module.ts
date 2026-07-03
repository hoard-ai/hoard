import { Module } from '@nestjs/common';

import { KnowledgeGraphConfigModule } from '@/config/knowledge-graph';

import { EdgeExtractionService } from './edge-extraction.service';
import { NodeExtractionService } from './node-extraction.service';

@Module({
  imports: [KnowledgeGraphConfigModule],
  providers: [NodeExtractionService, EdgeExtractionService],
  exports: [NodeExtractionService, EdgeExtractionService],
})
export class ExtractionModule {}
