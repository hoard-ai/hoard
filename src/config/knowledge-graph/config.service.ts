import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class KnowledgeGraphConfigService {
  constructor(private readonly configService: ConfigService) {}

  get memoryBackpressureConcurrencyLimit(): number {
    return this.configService.get<number>(
      'knowledgeGraph.memoryBackpressureConcurrencyLimit',
    )!;
  }

  get communityUpdateJobConcurrency(): number {
    return this.configService.get<number>(
      'knowledgeGraph.communityUpdateJobConcurrency',
    )!;
  }
}
