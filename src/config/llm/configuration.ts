import { registerAs } from '@nestjs/config';
import { z } from 'zod';

const envSchema = z.object({
  PLATFORM_MODEL_ENABLED: z
    .string()
    .default('false')
    .transform((val) => val === 'true' || val === 'TRUE'),
  GEMINI_API_KEY: z.string().optional(),
  LLM_PLATFORM_MODEL: z.string().default('gemini-3.0-flash'),
  // Max concurrent in-flight LLM generations against the shared platform key
  // (one pool across all PLATFORM users).
  LLM_PLATFORM_MAX_CONCURRENCY: z.coerce.number().int().positive().default(10),
});

export default registerAs('llm', () => {
  const env = envSchema.parse(process.env);
  return {
    platformModelEnabled: env.PLATFORM_MODEL_ENABLED,
    geminiApiKey: env.GEMINI_API_KEY,
    platformModel: env.LLM_PLATFORM_MODEL,
    platformMaxConcurrency: env.LLM_PLATFORM_MAX_CONCURRENCY,
  };
});
