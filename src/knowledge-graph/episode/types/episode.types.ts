import { z } from 'zod';

import { UuidSchema } from '@/common/schemas';

import {
  EntityEdgeSchema,
  EntityNodeSchema,
  EpisodicEdgeSchema,
  EpisodicNodeSchema,
} from '../../models';
import {
  EpisodeType,
  NodeLabelSchema,
  NodeNameSchema,
  RelationshipTypeSchema,
} from '../../types';

// Constants

export const PREVIOUS_EPISODES_WINDOW = 20;

// ─── Per-ingestion-path input schemas ──────────────────────────────────────
//
// Each ingestion path (text, message, ...) gets its own flat input schema so
// agents bind a tool with no oneOf branches to navigate. The service
// normalises both into a single internal shape before running the pipeline.

const MessageTurnSchema = z.object({
  speakerName: z
    .string()
    .min(1)
    .describe(
      'Name of the person who produced this turn. Use a real name when known; avoid generic role labels like "User"/"Assistant"/"Bot".',
    ),
  message: z
    .string()
    .min(1)
    .describe('Verbatim text of the turn, exactly as the speaker produced it.'),
});

const EpisodeInputBaseSchema = z.object({
  graphId: UuidSchema.describe('ID of the knowledge graph this episode belongs to.'),
  name: NodeNameSchema.describe(
    'Short identifier for the episode, used for logging and prompt context.',
  ),
  sourceDescription: z
    .string()
    .min(1)
    .describe(
      'Identifier of the ingestion source (e.g. "RememberNow UI", "Slack importer v1"). Surfaces in extraction prompts as supplementary context.',
    ),
  referenceTime: z.iso
    .datetime()
    .default(() => new Date().toISOString())
    .describe(
      'ISO 8601 timestamp with Z suffix marking when the content occurred. Used to resolve relative time references during extraction. Defaults to now.',
    ),
  id: UuidSchema.optional().describe(
    'Optional explicit episode ID. Omit to have one generated.',
  ),
  sagaId: UuidSchema.optional().describe(
    'Optional ID of a saga (topic/thread) this episode contributes to.',
  ),
});

const TextEpisodeInputSchema = EpisodeInputBaseSchema.extend({
  content: z
    .string()
    .min(1)
    .describe('Free-form text. Narration, notes, or any single-string payload.'),
});

const MessageEpisodeInputSchema = EpisodeInputBaseSchema.extend({
  content: z
    .array(MessageTurnSchema)
    .min(1)
    .describe('Conversation turns in chronological order.'),
});

// prepareChunks in episode/content-chunking.ts relies on this refine
const JsonEpisodeInputSchema = EpisodeInputBaseSchema.extend({
  content: z
    .string()
    .min(1)
    .refine(
      (s) => {
        try {
          JSON.parse(s);
          return true;
        } catch {
          return false;
        }
      },
      { message: 'content must be valid JSON' },
    )
    .describe(
      'JSON-stringified payload. Chunked per array element when the top-level value is an array.',
    ),
});

// ─── Type-map schemas ──────────────────────────────────────────────────────

const AttributeSchema = z
  .instanceof(z.ZodType)
  .refine((s) => s instanceof z.ZodObject && Object.keys(s.shape).length > 0, {
    message: 'Attribute schema must be a non-empty z.object',
  });

export const EntityTypeMapSchema = z.record(
  NodeLabelSchema,
  z.object({
    description: z.string(),
    schema: AttributeSchema,
  }),
);
export const EdgeTypeMapSchema = z.record(
  RelationshipTypeSchema,
  z.object({
    description: z.string(),
    schema: AttributeSchema,
  }),
);

export const EdgeTypeMappingsSchema = z.map(
  z.tuple([NodeLabelSchema, NodeLabelSchema]),
  z.array(RelationshipTypeSchema),
);

// ─── Public addEpisodes options (one per ingestion path) ───────────────────

const AddEpisodesOptionsBaseSchema = z.object({
  userId: UuidSchema,
  entityTypes: EntityTypeMapSchema.optional(),
  edgeTypes: EdgeTypeMapSchema.optional(),
  edgeTypeMappings: EdgeTypeMappingsSchema.optional(),
  excludedEntityTypes: z.array(NodeLabelSchema).optional(),
  customInstructions: z.string().optional(),
  updateCommunities: z.boolean().optional(),
});

export const AddTextEpisodesOptionsSchema = AddEpisodesOptionsBaseSchema.extend({
  episodes: z.array(TextEpisodeInputSchema).min(1),
});

export const AddMessageEpisodesOptionsSchema = AddEpisodesOptionsBaseSchema.extend({
  episodes: z.array(MessageEpisodeInputSchema).min(1),
});

export const AddJsonEpisodesOptionsSchema = AddEpisodesOptionsBaseSchema.extend({
  episodes: z.array(JsonEpisodeInputSchema).min(1),
});

// ─── Internal normalised shape (after content-flatten + source attached) ───
//
// Not exposed to callers; produced by the public methods and consumed by
// the private pipeline impl.

const NormalizedEpisodeSchema = EpisodeInputBaseSchema.extend({
  source: z.enum(EpisodeType),
  content: z.string().min(1),
  referenceTime: z.date(),
});

export const NormalizedAddEpisodeOptionsSchema = AddEpisodesOptionsBaseSchema.extend({
  episodes: z.array(NormalizedEpisodeSchema).min(1),
});

export const AddEpisodeResultSchema = z.object({
  nodes: z.array(EntityNodeSchema),
  edges: z.array(EntityEdgeSchema),
  invalidatedEdges: z.array(EntityEdgeSchema),
  episodicEdges: z.array(EpisodicEdgeSchema),
  episode: EpisodicNodeSchema,
});

// ─── Types ─────────────────────────────────────────────────────────────────

export type EntityTypeMap = z.infer<typeof EntityTypeMapSchema>;
export type EdgeTypeMap = z.infer<typeof EdgeTypeMapSchema>;
export type EdgeTypeMappings = z.infer<typeof EdgeTypeMappingsSchema>;

export type AddTextEpisodesOptionsInput = z.input<typeof AddTextEpisodesOptionsSchema>;
export type AddMessageEpisodesOptionsInput = z.input<
  typeof AddMessageEpisodesOptionsSchema
>;
export type AddJsonEpisodesOptionsInput = z.input<typeof AddJsonEpisodesOptionsSchema>;

export type NormalizedAddEpisodeOptions = z.infer<
  typeof NormalizedAddEpisodeOptionsSchema
>;
export type AddEpisodeResult = z.infer<typeof AddEpisodeResultSchema>;
