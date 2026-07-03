import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

import { KnowledgeGraphConfigService } from './config.service';
import configuration from './configuration';

@Module({
  imports: [ConfigModule.forFeature(configuration)],
  providers: [KnowledgeGraphConfigService],
  exports: [KnowledgeGraphConfigService],
})
export class KnowledgeGraphConfigModule {}
