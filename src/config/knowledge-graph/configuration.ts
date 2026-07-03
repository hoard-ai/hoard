import { registerAs } from '@nestjs/config';
import { z } from 'zod';

const envSchema = z.object({
  // Max concurrent tasks per in-process fan-out. This is memory backpressure
  // (bounds how many large prompts are materialized at once), NOT the LLM rate
  // limit - that is enforced per-credential at the model layer.
  MEMORY_BACKPRESSURE_CONCURRENCY_LIMIT: z.coerce.number().int().positive().default(10),
  // How many community-update jobs a worker processes at once. Provider rate is
  // bounded per-credential by the semaphores, so this only controls job
  // parallelism (each job is a sequential per-entity loop).
  COMMUNITY_UPDATE_JOB_CONCURRENCY: z.coerce.number().int().positive().default(10),
});

export default registerAs('knowledgeGraph', () => {
  const env = envSchema.parse(process.env);
  return {
    memoryBackpressureConcurrencyLimit: env.MEMORY_BACKPRESSURE_CONCURRENCY_LIMIT,
    communityUpdateJobConcurrency: env.COMMUNITY_UPDATE_JOB_CONCURRENCY,
  };
});
