import { Module } from '@nestjs/common';

import { EmbeddingConfigModule } from '@/config/embedding';
import { RateLimitModule } from '@/providers/rate-limit';

import { EmbeddingService } from './embedding.service';

@Module({
  imports: [EmbeddingConfigModule, RateLimitModule],
  providers: [EmbeddingService],
  exports: [EmbeddingService],
})
export class EmbeddingModule {}
