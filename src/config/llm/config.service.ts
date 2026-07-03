import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class LlmConfigService {
  constructor(private readonly configService: ConfigService) {}

  get platformModelEnabled(): boolean {
    return this.configService.get<boolean>('llm.platformModelEnabled')!;
  }

  get geminiApiKey(): string | undefined {
    return this.configService.get<string>('llm.geminiApiKey');
  }

  get platformModel(): string {
    return this.configService.get<string>('llm.platformModel')!;
  }

  get platformMaxConcurrency(): number {
    return this.configService.get<number>('llm.platformMaxConcurrency')!;
  }
}
