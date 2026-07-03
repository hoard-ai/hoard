import { createZodDto } from 'nestjs-zod';
import { z } from 'zod';

import { LlmProvider } from '@generated/prisma/client';

const PrismaLlmProvider = Object.values(LlmProvider) as [LlmProvider, ...LlmProvider[]];

export const LlmProviderSchema = z.enum(PrismaLlmProvider).meta({ id: 'LlmProvider' });

//  Adding a new provider (e.g., OpenAI) requires touching 5 locations:
//  1. Prisma enum
//  2. Base schema in DTO (BaseOpenAISchema)
//  3. Save schema in DTO (extend with apiKey)
//  4. Response schema in DTO (extend with response fields)
//  5. Factory switch case

// ── Base schemas (provider-specific params, no apiKey) ──────────────

const modelSchema = z.string().min(1);

// Per-model cap on concurrent in-flight LLM generations - the rate-limit pool
// size for this credential. Unset falls back to DEFAULT_MODEL_CONCURRENCY.
const concurrencySchema = z.int().positive().optional();

const BaseModelSchema = z.object({
  model: modelSchema,
  concurrency: concurrencySchema,
});

// https://reference.langchain.com/javascript/langchain-anthropic/AnthropicInput
const BaseAnthropicSchema = BaseModelSchema.extend({
  provider: z.literal(LlmProvider.ANTHROPIC),
  // https://docs.anthropic.com/claude/docs/models-overview
  temperature: z.number().min(0).max(1).optional(),
  topK: z.number().int().positive().optional(),
});

// https://reference.langchain.com/javascript/langchain-google-genai/GoogleGenerativeAIChatInput
const BaseGoogleGeminiSchema = BaseModelSchema.extend({
  provider: z.literal(LlmProvider.GOOGLE_GEMINI),
  // https://ai.google.dev/gemini-api/docs/models
  temperature: z.number().min(0).max(2).optional(),
  topK: z.number().int().positive().optional(),
});

// ── Save schemas (base + apiKey) ────────────────────────────────────

// WARNING: Changing this could break data validation in getActiveModel
const apiKeySchema = z.string().min(1).optional();

export const AnthropicConfigSchema = BaseAnthropicSchema.extend({
  apiKey: apiKeySchema,
});

export const GoogleGeminiConfigSchema = BaseGoogleGeminiSchema.extend({
  apiKey: apiKeySchema,
});

export const SaveLlmConfigSchema = z
  .discriminatedUnion('provider', [AnthropicConfigSchema, GoogleGeminiConfigSchema])
  .meta({ id: 'SaveLlmConfig' });

// ── Response schemas (base with model optional + response fields) ───

const configResponseFields = {
  model: z.string().optional(),
  hasApiKey: z.boolean(),
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
};

const AnthropicConfigResponseSchema = BaseAnthropicSchema.extend(configResponseFields);

const GoogleGeminiConfigResponseSchema =
  BaseGoogleGeminiSchema.extend(configResponseFields);

export const LlmConfigResponseSchema = z
  .discriminatedUnion('provider', [
    AnthropicConfigResponseSchema,
    GoogleGeminiConfigResponseSchema,
  ])
  .meta({ id: 'LlmConfigResponse' });

export const LlmProvidersListSchema = z
  .object({
    activeProvider: LlmProviderSchema.nullable(),
    providers: z.array(LlmConfigResponseSchema),
  })
  .meta({ id: 'LlmProvidersList' });

export const TestConfigResponseSchema = z
  .object({
    provider: LlmProviderSchema,
    success: z.boolean(),
    message: z.string(),
    responseTime: z.number().optional(),
  })
  .meta({ id: 'TestConfigResponse' });

export const UserLlmProviderSchema = LlmProviderSchema.exclude([
  LlmProvider.PLATFORM,
]).meta({
  id: 'UserLlmProvider',
});

export const ProviderParamSchema = z
  .object({
    provider: UserLlmProviderSchema,
  })
  .meta({ id: 'ProviderParam' });

export const SetActiveProviderSchema = z
  .object({
    provider: LlmProviderSchema.nullable(),
  })
  .meta({ id: 'SetActiveProvider' });

export const ActiveProviderResponseSchema = z
  .object({
    activeProvider: LlmProviderSchema.nullable(),
  })
  .meta({ id: 'ActiveProviderResponse' });

// DTO classes

// const + type pattern: createZodDto can't be `extends`-ed with a discriminated union (TS2509)
// Object.defineProperty fixes the class name so Swagger registers schemas under the correct name
// instead of the internal "AugmentedZodDto" from nestjs-zod.
export const SaveLlmConfigDto = createZodDto(SaveLlmConfigSchema);
Object.defineProperty(SaveLlmConfigDto, 'name', { value: 'SaveLlmConfig' });
export type SaveLlmConfigDto = InstanceType<typeof SaveLlmConfigDto>;

export const LlmConfigResponseDto = createZodDto(LlmConfigResponseSchema);
Object.defineProperty(LlmConfigResponseDto, 'name', {
  value: 'LlmConfigResponse',
});
export type LlmConfigResponseDto = InstanceType<typeof LlmConfigResponseDto>;

export class LlmProvidersListDto extends createZodDto(LlmProvidersListSchema) {}
export class TestConfigResponseDto extends createZodDto(TestConfigResponseSchema) {}
export class ProviderParamDto extends createZodDto(ProviderParamSchema) {}
export class SetActiveProviderDto extends createZodDto(SetActiveProviderSchema) {}
export class ActiveProviderResponseDto extends createZodDto(
  ActiveProviderResponseSchema,
) {}

// Types
export type SaveLlmConfig = z.infer<typeof SaveLlmConfigSchema>;
export type LlmConfigResponse = z.infer<typeof LlmConfigResponseSchema>;
export type LlmProvidersList = z.infer<typeof LlmProvidersListSchema>;
export type TestConfigResponse = z.infer<typeof TestConfigResponseSchema>;
export type SetActiveProvider = z.infer<typeof SetActiveProviderSchema>;
export type ActiveProviderResponse = z.infer<typeof ActiveProviderResponseSchema>;
