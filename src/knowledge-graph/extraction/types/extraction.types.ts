import { z } from 'zod';

import { type Uuid, UuidSchema } from '@/common/schemas';

import type { EntityEdge, EntityNode, EpisodicNode } from '../../models';

// TODO: fixed size assumes small chunks - make adaptive on the summary prompt's
// token budget (the episode-text chunk union + node payload) so large chunks
// don't overflow the context.
export const MAX_NODES_PER_SUMMARY_BATCH = 30;

export const EntityCorefDescriptorSchema = z.object({
  identifyingDescription: z
    .string()
    .trim()
    .min(1)
    .describe(
      'Standalone descriptor identifying this entity where its name may not ' +
        'appear: its role or type plus the most distinctive facts stated here (e.g. ' +
        '"neurologist; lead author of the migraine study"). Written so a later ' +
        'reader holding only this descriptor could tell which entity a pronoun ' +
        'refers to. State gender only when the text itself establishes it (a ' +
        'pronoun, a stated fact) - never inferred from the name.',
    ),
  aliases: z
    .array(z.string().trim().min(1))
    .default([])
    .describe(
      'Distinctive surface forms this text uses for the entity: name variants and ' +
        'definite descriptions (e.g. "Dr. Osei", "the lead researcher"). NEVER ' +
        'bare pronouns (he, she, they, it, this, that). Omit when none present.',
    ),
  referredToAsPronouns: z
    .array(z.string().trim().min(1))
    .default([])
    .describe(
      'Pronouns this text itself uses for the entity, in subject form (e.g. "he", ' +
        '"she", "it", "they"). Observed usage only - never inferred from the entity ' +
        'name or type. Omit when none present.',
    ),
});
export type EntityCorefDescriptor = z.infer<typeof EntityCorefDescriptorSchema>;

// TODO: Emit precise line numbers for unresolved references when
// offset-preserving chunking is achieved
export const UnresolvedReferenceSchema = z.object({
  surfaceForm: z
    .string()
    .trim()
    .min(1)
    .describe(
      'The referring expression that could not be tied to any extracted entity (e.g. "she", "the manager").',
    ),
  locatingQuote: z
    .string()
    .trim()
    .min(1)
    .describe(
      'Shortest quote around the reference, sufficient to locate it in the text.',
    ),
});
export type UnresolvedReference = z.infer<typeof UnresolvedReferenceSchema>;

// id + chunk are assigned service-side (never by the model), so a later chunk's
// edge extraction can claim the reference by id.
export const TrackedUnresolvedReferenceSchema = UnresolvedReferenceSchema.extend({
  id: UuidSchema,
  sourceChunkIndex: z.int().nonnegative(),
});
export type TrackedUnresolvedReference = z.infer<typeof TrackedUnresolvedReferenceSchema>;

// The model resolved a coreference (e.g. he) with this entity. The decision cannot
// be relitigated in further prompts. `locatingQuote` anchors the binding to its
// occurrence so same-form bindings to different entities stay distinguishable.
export const CommittedCorefBindingSchema = z.object({
  surfaceForm: z.string().trim().min(1),
  boundNodeId: UuidSchema,
  sourceChunkIndex: z.int().nonnegative(),
  locatingQuote: z.string().trim().min(1),
  // null when edge extraction resolved an occurrence node extraction never flagged.
  resolvedUnresolvedReferenceId: UuidSchema.nullable(),
});
export type CommittedCorefBinding = z.infer<typeof CommittedCorefBindingSchema>;

export type ExtractNodesResult = {
  nodes: EntityNode[];
  // Chunk indices each node was extracted from (unioned across chunks).
  chunkIndicesByExtractedId: Map<Uuid, Set<number>>;
  // Coref descriptors keyed by extracted node id (always populated).
  corefByExtractedId: Map<Uuid, EntityCorefDescriptor>;
  // Referring expressions the chunk(s) could not resolve locally.
  unresolvedReferences: TrackedUnresolvedReference[];
};

export type NodeEpisodeContext = Map<
  Uuid,
  {
    episode: EpisodicNode;
    previousEpisodes: EpisodicNode[];
    // The episode's chunks + this node's chunk indices, so attribute/summary
    // episode text is scoped to where the node was discussed (selectChunkText).
    chunks: string[];
    sourceChunkIndices: Set<number>;
    // This episode's committed bindings; filtered by the node's chunks at the call site.
    committedCorefBindings: CommittedCorefBinding[];
  }
>;

export type ExtractEdgesResult = {
  edges: EntityEdge[];
  // Chunk index each edge was extracted from (stays a singleton - dedup keeps
  // each edge id's own origin entry and never unions, see EdgeChunkSources).
  chunkIndicesByEdgeId: Map<Uuid, Set<number>>;
  // Bindings extract-edges committed to (empty when the flag is off).
  committedCorefBindings: CommittedCorefBinding[];
};
