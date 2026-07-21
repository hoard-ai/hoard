import { z } from 'zod';

import { UuidSchema } from '@/common/schemas';

export enum EpisodeType {
  message = 'message',
  json = 'json',
  text = 'text',
}

export const EpisodeTypeSchema = z.enum(EpisodeType);

export const NodeNameSchema = z.string().trim().min(1).brand<'NodeName'>();

export const NodeLabelSchema = z
  .string()
  .regex(
    /^[A-Za-z_][A-Za-z0-9_]*$/,
    'node label must start with a letter or underscore and contain only alphanumeric characters or underscores',
  )
  .brand<'NodeLabel'>();

export const NodeLabelsSchema = z.array(NodeLabelSchema).min(1);

export const RelationshipTypeSchema = z
  .string()
  .regex(
    /^[A-Z0-9]+(_[A-Z0-9]+)*$/,
    'relationship type must be SCREAMING_SNAKE_CASE (uppercase letters, digits, single underscores between segments)',
  )
  .brand<'RelationshipType'>();

export const SearchByTextParamsSchema = z.object({
  query: z.string(),
  graphIds: z.array(UuidSchema),
  limit: z.number().int().positive(),
});

export const SearchBySimilarityParamsSchema = z.object({
  embedding: z.array(z.number()),
  graphIds: z.array(UuidSchema),
  limit: z.number().int().positive(),
  minScore: z.number().default(0),
});

export const SearchByBfsParamsSchema = z.object({
  originNodeIds: z.array(UuidSchema),
  graphIds: z.array(UuidSchema),
  limit: z.number().int().positive(),
  maxDepth: z.number().int().positive().optional(),
});

export const RetrieveEpisodesParamsSchema = z.object({
  graphIds: z.array(UuidSchema),
  referenceTime: z.date().default(() => new Date()),
  lastN: z.number().int().positive().default(10),
  source: EpisodeTypeSchema.optional(),
  sagaId: UuidSchema.optional(),
});

export type NodeLabel = z.infer<typeof NodeLabelSchema>;
export type NodeLabels = z.infer<typeof NodeLabelsSchema>;
export type NodeName = z.infer<typeof NodeNameSchema>;
export type RelationshipType = z.infer<typeof RelationshipTypeSchema>;
export type SearchByTextParams = z.infer<typeof SearchByTextParamsSchema>;
export type SearchBySimilarityParams = z.infer<typeof SearchBySimilarityParamsSchema>;
export type SearchByBfsParams = z.infer<typeof SearchByBfsParamsSchema>;
export type RetrieveEpisodesParamsInput = z.input<typeof RetrieveEpisodesParamsSchema>;
export type RetrieveEpisodesParams = z.infer<typeof RetrieveEpisodesParamsSchema>;
