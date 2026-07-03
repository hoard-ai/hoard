import { registerAs } from '@nestjs/config';
import { z } from 'zod';

const envSchema = z.object({
  EMBEDDING_ENABLED: z
    .string()
    .default('false')
    .transform((val) => val === 'true' || val === 'TRUE'),
  GOOGLE_EMBEDDING_API_KEY: z.string().optional(),
  GOOGLE_EMBEDDING_MODEL: z.string().default('gemini-embedding-2'),
  GOOGLE_EMBEDDING_DIMENSIONS: z.coerce.number().int().positive().default(768),
  EMBEDDING_CONCURRENCY_LIMIT: z.coerce.number().int().positive().default(10),
});

export default registerAs('embedding', () => {
  const env = envSchema.parse(process.env);
  return {
    embeddingEnabled: env.EMBEDDING_ENABLED,
    googleApiKey: env.GOOGLE_EMBEDDING_API_KEY,
    googleModel: env.GOOGLE_EMBEDDING_MODEL,
    dimensions: env.GOOGLE_EMBEDDING_DIMENSIONS,
    concurrencyLimit: env.EMBEDDING_CONCURRENCY_LIMIT,
  };
});
