import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class EmbeddingConfigService {
  constructor(private readonly configService: ConfigService) {}

  get embeddingEnabled(): boolean {
    return this.configService.get<boolean>('embedding.embeddingEnabled')!;
  }

  get googleApiKey(): string | undefined {
    return this.configService.get<string>('embedding.googleApiKey');
  }

  get googleModel(): string {
    return this.configService.get<string>('embedding.googleModel')!;
  }

  get dimensions(): number {
    return this.configService.get<number>('embedding.dimensions')!;
  }

  get concurrencyLimit(): number {
    return this.configService.get<number>('embedding.concurrencyLimit')!;
  }
}
