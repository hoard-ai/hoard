import { randomUUID } from 'node:crypto';

import { z } from 'zod';

import { Uuid, UuidSchema } from '@/common/schemas';

import { EpisodeType, NodeLabelSchema, NodeName, NodeNameSchema } from '../types';

// Schemas

export const NodeBaseSchema = z.object({
  id: UuidSchema,
  name: NodeNameSchema,
  graphId: UuidSchema,
  labels: z.array(NodeLabelSchema),
  createdAt: z.date(),
});

export const EntityNodeSchema = NodeBaseSchema.extend({
  nameEmbedding: z.array(z.number()).nullable().default(null),
  summary: z.string().default(''),
  attributes: z.record(z.string(), z.unknown()).default({}),
});

export const EpisodicNodeSchema = NodeBaseSchema.extend({
  source: z.enum(EpisodeType),
  sourceDescription: z.string().min(1),
  content: z.string(),
  validAt: z.date(),
  sagaId: UuidSchema.nullable().default(null),
  // entityEdges omitted: upstream's Neo4j denormalization of "facts this
  // episode contributed to" lives on `entity_edges.episodes` here. For the
  // upstream semantic, query `EntityEdge WHERE thisEpisode = ANY(episodes)`.

  // episodeMetadata omitted: upstream declares a "customer-defined metadata
  // for filtering" field but never writes or reads it. Add only if a
  // concrete filterable-metadata requirement materializes.
});

// Communities are auto-generated clusters of entity nodes; they don't carry
// user-defined labels (always implicitly 'Community') and have no `attributes`.
//
// member_signature lives in the DB but is intentionally absent here: a
// BEFORE INSERT/UPDATE trigger derives it from member_ids + entity_nodes.summary,
// so it's never a value the application produces, passes, or compares in TS.
export const CommunitySchema = z.object({
  id: UuidSchema,
  graphId: UuidSchema,
  name: NodeNameSchema,
  summary: z.string().default(''),
  nameEmbedding: z.array(z.number()).nullable().default(null),
  memberIds: z.array(UuidSchema).default([]),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export const SagaSchema = NodeBaseSchema.extend({
  summary: z.string().default(''),
  lastSummarizedAt: z.date().nullable().default(null),
  // firstEpisodeUuid / lastEpisodeUuid omitted: upstream persists them but
  // never reads either. The queries they'd optimize (start/end of saga) are
  // cheap via episodic_nodes.saga_id + ORDER BY valid_at.
});

// Types

export type NodeBase = z.infer<typeof NodeBaseSchema>;
export type EntityNode = z.infer<typeof EntityNodeSchema>;
export type EpisodicNode = z.infer<typeof EpisodicNodeSchema>;
export type Community = z.infer<typeof CommunitySchema>;
export type Saga = z.infer<typeof SagaSchema>;

// Factories

export function createNodeDefaults(): Omit<NodeBase, 'name' | 'graphId'> {
  return {
    id: UuidSchema.parse(randomUUID()),
    labels: [],
    createdAt: new Date(),
  };
}

export function createEntityNode(
  partial: Partial<EntityNode> & { name: NodeName; graphId: Uuid },
): EntityNode {
  return EntityNodeSchema.parse({
    ...createNodeDefaults(),
    labels: [NodeLabelSchema.parse('Entity')],
    ...partial,
  });
}

export function createEpisodicNode(
  partial: Partial<EpisodicNode> & {
    name: NodeName;
    graphId: Uuid;
    content: string;
    validAt: Date;
  },
): EpisodicNode {
  return EpisodicNodeSchema.parse({
    ...createNodeDefaults(),
    labels: [NodeLabelSchema.parse('Episodic')],
    source: EpisodeType.text,
    ...partial,
  });
}

export function createCommunity(
  partial: Partial<Community> & {
    name: NodeName;
    graphId: Uuid;
    memberIds: Uuid[];
  },
): Community {
  const now = new Date();
  return CommunitySchema.parse({
    id: UuidSchema.parse(randomUUID()),
    createdAt: now,
    updatedAt: now,
    ...partial,
  });
}

export function createSaga(
  partial: Partial<Saga> & { name: NodeName; graphId: Uuid },
): Saga {
  return SagaSchema.parse({
    ...createNodeDefaults(),
    labels: [NodeLabelSchema.parse('Saga')],
    ...partial,
  });
}
