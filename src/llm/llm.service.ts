import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';

import { LlmConfig, LlmProvider } from '@generated/prisma/client';

import { Uuid } from '@/common/schemas';
import { LlmConfigService } from '@/config/llm';
import { recordSpanEvent } from '@/observability';
import { CryptoService } from '@/providers/crypto';
import { PrismaService } from '@/providers/database/postgres';
import { RateLimiterService } from '@/providers/rate-limit';

import {
  ActiveProviderResponse,
  LlmConfigResponse,
  LlmConfigResponseSchema,
  LlmProvidersList,
  SaveLlmConfig,
  SaveLlmConfigSchema,
  TestConfigResponse,
} from './dto';
import { LlmFactoryService } from './factory/llm-factory.service';
import { DEFAULT_MODEL_CONCURRENCY, GetActiveModelOptions } from './types';

@Injectable()
export class LlmService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly factory: LlmFactoryService,
    private readonly llmConfig: LlmConfigService,
    private readonly crypto: CryptoService,
    private readonly rateLimiter: RateLimiterService,
  ) {}

  private parseConfigRow(row: LlmConfig): SaveLlmConfig {
    // TODO: The encrypted ciphertext key is stored in apiKey
    // This can introduce subtle auth errors
    return SaveLlmConfigSchema.parse({
      ...(row.config as Record<string, unknown>),
      provider: row.provider,
    });
  }

  private toConfigResponse(row: LlmConfig): LlmConfigResponse {
    const config = this.parseConfigRow(row);

    return LlmConfigResponseSchema.parse({
      provider: config.provider,
      model: config.model,
      concurrency: config.concurrency,
      temperature: config.temperature,
      topK: config.topK,
      hasApiKey: !!config.apiKey,
      createdAt: row.createdAt.toISOString(),
      updatedAt: row.updatedAt.toISOString(),
    });
  }

  async listProviders(userId: Uuid): Promise<LlmProvidersList> {
    const [user, configs] = await Promise.all([
      this.prisma.user.findUniqueOrThrow({
        where: { id: userId },
        select: { activeLlmProvider: true },
      }),
      this.prisma.llmConfig.findMany({
        where: { userId },
      }),
    ]);
    const configMap = new Map(configs.map((c) => [c.provider, c]));
    const providers: LlmConfigResponse[] = [];

    for (const provider of Object.values(LlmProvider)) {
      if (provider === LlmProvider.PLATFORM) continue;
      const config = configMap.get(provider);

      providers.push(
        config
          ? this.toConfigResponse(config)
          : LlmConfigResponseSchema.parse({ provider, hasApiKey: false }),
      );
    }
    return { activeProvider: user.activeLlmProvider, providers };
  }

  async getProviderConfig(
    userId: Uuid,
    provider: LlmProvider,
  ): Promise<LlmConfigResponse> {
    if (provider === LlmProvider.PLATFORM) {
      throw new BadRequestException('Platform provider is server-managed');
    }
    const config = await this.prisma.llmConfig.findUnique({
      where: { userId_provider: { userId, provider } },
    });

    if (!config) {
      return LlmConfigResponseSchema.parse({ provider, hasApiKey: false });
    }
    return this.toConfigResponse(config);
  }

  async saveProviderConfig(
    userId: Uuid,
    config: SaveLlmConfig,
  ): Promise<LlmConfigResponse> {
    const existingRow = await this.prisma.llmConfig.findUnique({
      where: { userId_provider: { userId, provider: config.provider } },
    });
    const existing = existingRow ? this.parseConfigRow(existingRow) : null;

    // Encrypt new plaintext key, or retain existing encrypted key
    const apiKey = config.apiKey ? this.crypto.encrypt(config.apiKey) : existing?.apiKey;

    // Merge: existing as base, incoming overrides (strip provider + apiKey)
    const {
      provider,
      apiKey: _,
      ...merged
    } = existing ? { ...existing, ...config } : config;

    const configJson = apiKey ? { ...merged, apiKey } : merged;

    // Validate merged config before persisting. Defensive
    SaveLlmConfigSchema.parse({ ...configJson, provider });

    const result = await this.prisma.llmConfig.upsert({
      where: { userId_provider: { userId, provider } },
      create: { userId, provider, config: configJson },
      update: { config: configJson },
    });
    return this.toConfigResponse(result);
  }

  async deleteProviderConfig(userId: Uuid, provider: LlmProvider): Promise<void> {
    if (provider === LlmProvider.PLATFORM) {
      throw new BadRequestException('Platform provider is server-managed');
    }
    await this.prisma.$transaction(async (tx) => {
      const config = await tx.llmConfig.findUnique({
        where: { userId_provider: { userId, provider } },
      });

      if (!config) {
        throw new NotFoundException('Provider config not found');
      }

      await tx.llmConfig.delete({
        where: { id: config.id },
      });

      // Unset active provider if it was the deleted one
      await tx.user.updateMany({
        where: { id: userId, activeLlmProvider: provider },
        data: { activeLlmProvider: null },
      });
    });
  }

  async setActiveProvider(
    userId: Uuid,
    provider: LlmProvider | null,
  ): Promise<ActiveProviderResponse> {
    if (provider === null) {
      await this.prisma.user.update({
        where: { id: userId },
        data: { activeLlmProvider: null },
      });
      return { activeProvider: null };
    }

    if (provider === LlmProvider.PLATFORM) {
      if (!this.llmConfig.platformModelEnabled) {
        throw new BadRequestException('Platform model is not enabled');
      }
    } else {
      const configRow = await this.prisma.llmConfig.findUnique({
        where: { userId_provider: { userId, provider } },
      });
      if (!configRow) {
        throw new BadRequestException(
          `Provider ${provider} is not configured. Save a config first.`,
        );
      }
      const config = this.parseConfigRow(configRow);
      if (!config.apiKey) {
        throw new BadRequestException(`Provider ${provider} is missing an API key`);
      }
    }

    await this.prisma.user.update({
      where: { id: userId },
      data: { activeLlmProvider: provider },
    });
    return { activeProvider: provider };
  }

  async testProviderConfig(
    userId: Uuid,
    provider: LlmProvider,
  ): Promise<TestConfigResponse> {
    if (provider === LlmProvider.PLATFORM) {
      throw new BadRequestException('Platform model is not testable');
    }
    const model = await this.getActiveModel(userId, {
      providerOverride: provider,
      rateLimited: false,
    });
    const start = Date.now();

    try {
      await model.invoke('Say "hello" in one word.', { timeout: 4000 });
      const responseTime = Date.now() - start;

      return {
        provider,
        success: true,
        message: 'Connection successful',
        responseTime,
      };
    } catch (error) {
      const responseTime = Date.now() - start;
      return {
        provider,
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        responseTime,
      };
    }
  }

  /**
   * Resolves the user's active provider into a ready-to-use chat model.
   *
   * By default (`rateLimited` defaults to `true`) the returned model's
   * `_generate` is gated by the per-credential concurrency limiter; pass
   * `rateLimited: false` for latency-sensitive interactive paths (e.g. the agent
   * chat loop) that must not queue. Bulk work like the cross-encoder reranker
   * stays rate limited. The shared PLATFORM key is always rate limited - it cannot
   * bypass its pool. `abortSignal` scopes cancellation to this model, and
   * `providerOverride` bypasses the user's `activeLlmProvider` lookup.
   */
  async getActiveModel(
    userId: Uuid,
    opts: GetActiveModelOptions = {},
  ): Promise<BaseChatModel> {
    const { providerOverride, rateLimited = true, abortSignal } = opts;
    const provider =
      providerOverride ??
      (
        await this.prisma.user.findUniqueOrThrow({
          where: { id: userId },
          select: { activeLlmProvider: true },
        })
      ).activeLlmProvider;

    if (!provider) {
      throw new BadRequestException('No active LLM provider is set');
    }
    if (provider === LlmProvider.PLATFORM) {
      const model = this.factory.createPlatformModel();
      return this.wrapRateLimited(
        model,
        'platform',
        this.llmConfig.platformMaxConcurrency,
        abortSignal,
      );
    }

    const configRow = await this.prisma.llmConfig.findUnique({
      where: { userId_provider: { userId, provider } },
    });
    if (!configRow) {
      throw new NotFoundException(`No configuration found for provider ${provider}`);
    }

    const config = this.parseConfigRow(configRow);
    if (!config.apiKey) {
      throw new BadRequestException(`Provider ${provider} is missing an API key`);
    }

    const model = this.factory.createModel({
      ...config,
      apiKey: this.crypto.decrypt(config.apiKey),
    });
    if (!rateLimited) return model;

    // BYO key: rate limit scoped per credential
    const key = `user:${userId}:${provider}`;
    const limit = config.concurrency ?? DEFAULT_MODEL_CONCURRENCY;
    return this.wrapRateLimited(model, key, limit, abortSignal);
  }

  /**
   * Gates `model`'s `_generate` through the rate-limiter pool `key` (max `limit`
   * concurrent). The permit is acquired pre-dispatch, so a queued call holds no
   * HTTP connection and starts no request-timeout clock while it waits.
   * `abortSignal` rejects queued acquires and cancels in-flight requests.
   */
  private wrapRateLimited(
    model: BaseChatModel,
    key: string,
    limit: number,
    abortSignal?: AbortSignal,
  ): BaseChatModel {
    // TODO: Add request timeouts
    const holder = model as unknown as {
      _generate: (...args: GenerateParams) => GenerateReturn;
    };
    const orig = holder._generate.bind(model);

    // Own-property shadow of the prototype method: BaseChatModel routes both
    // invoke() and withStructuredOutput().invoke() through `this._generate`.
    holder._generate = (...args: GenerateParams): GenerateReturn => {
      const [messages, options, runManager] = args;
      const signal = mergeSignals(abortSignal, options.signal);
      const nextOptions = { ...options, signal } as GenerateParams[1];
      return this.rateLimiter.schedule(
        key,
        limit,
        signal,
        () => orig(messages, nextOptions, runManager),
        (waitMs) =>
          recordSpanEvent('ratelimit.admitted', {
            'ratelimit.key': key,
            'ratelimit.queue_wait_ms': waitMs,
          }),
      );
    };
    return model;
  }
}

type GenerateParams = Parameters<BaseChatModel['_generate']>;
type GenerateReturn = ReturnType<BaseChatModel['_generate']>;

function mergeSignals(a?: AbortSignal, b?: AbortSignal): AbortSignal | undefined {
  if (a && b) return AbortSignal.any([a, b]);
  return a ?? b;
}
