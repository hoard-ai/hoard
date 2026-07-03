import { Module } from '@nestjs/common';

import { RedisModule } from '@/providers/cache/redis';

import { RateLimiterService } from './rate-limiter.service';

@Module({
  imports: [RedisModule],
  providers: [RateLimiterService],
  exports: [RateLimiterService],
})
export class RateLimitModule {}
