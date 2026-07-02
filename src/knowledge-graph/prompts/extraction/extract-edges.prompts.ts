import { BaseMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { edgeTypeKey } from '@/knowledge-graph/episode/episode-utils';
import { EdgeTypeMap, EdgeTypeMappings } from '@/knowledge-graph/episode/types';
import { EntityNode, EpisodicNode } from '@/knowledge-graph/models';
import { RelationshipTypeSchema } from '@/knowledge-graph/types';
import type { Violation } from '@/llm';

import { formatCurrentEpisode, formatPreviousEpisodes } from '../text-utils';

// Schema

// TODO: Too much cognitive load for a model
const ExtractedEdgeSchema = z.object({
  sourceEntityIdx: z
    .int()
    .nonnegative()
    .describe('The 0-based id of the source entity from the ENTITIES list'),
  targetEntityIdx: z
    .int()
    .nonnegative()
    .describe('The 0-based id of the target entity from the ENTITIES list'),
  // TODO: This is where the model creates new relationship types. Unlike
  // extract-nodes (which forces the model to pick an entityTypeId from a
  // provided list), edges accept any SCREAMING_SNAKE_CASE name the model
  // derives from the predicate when no provided FACT TYPE matches. Kept soft
  // by design - tightening would require dropping support for novel relations.
  relationType: RelationshipTypeSchema.describe(
    'The type of relationship between the entities, in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)',
  ),
  fact: z
    .string()
    .describe(
      'A natural language description of the relationship between the entities, paraphrased from the source text',
    ),
  // TODO: Multi-episode extraction per prompt
  // episodeIndices: z
  //   .array(z.number())
  //   .default([0])
  //   .describe(
  //     'List of episode numbers (0-indexed) that this fact was derived from. When processing a single episode, this should be [0].',
  //   ),
});

export const ExtractedEdgesSchema = z.object({
  edges: z.array(ExtractedEdgeSchema).describe('List of extracted relationship facts'),
});

export type ExtractedEdgesOutput = z.infer<typeof ExtractedEdgesSchema>;

// Prompt builder

const SYSTEM_PROMPT = `You are an expert fact extractor. Extract factual relationships (edges)
between the given ENTITIES from the CURRENT EPISODE.

Primary goal:
Extract every clearly stated or unambiguously implied relationship between two DISTINCT entities
from the ENTITIES list that can be represented as an edge in a knowledge graph, paraphrased from
the source text with all specific details preserved.

Source rules:
- Only use facts grounded in the CURRENT EPISODE. The CURRENT EPISODE may contain multiple
episodes.
- Use PREVIOUS EPISODES only to disambiguate references or support continuity, never as a source
of new facts.

EXTRACTION RULES:

1. Entity Id Validation: 'sourceEntityIdx' and 'targetEntityIdx' MUST be the 'id' value of an entry
in the ENTITIES list provided in the human message.
   - CRITICAL: Using an id not present in the list will cause the edge to be rejected.
2. Each fact must involve two DISTINCT entities - 'sourceEntityIdx' and 'targetEntityIdx' NEVER
refer to the same entity (their ids MUST differ).
3. Prefer facts that involve two distinct entities from the ENTITIES list. When a sentence
describes a specific, concrete detail about a single entity (a brand name, a specific item, a
physical description, a quantity, a location, a named activity), do NOT drop it. Instead, look for
a second entity in the ENTITIES list that the detail relates to and form a proper edge (e.g.,
Entity -> OWNS -> item-entity, Entity -> LIVES_IN -> place-entity,
Entity -> HAS_ATTRIBUTE -> detail-entity). Only skip the fact when no second entity in the
ENTITIES list can anchor the detail.
   - BAD: "Alice feels happy" (vague single-entity state with no concrete detail - what is Alice happy about?)
   - GOOD: "Alice feels happy about Bob's promotion" -> Alice -> FEELS_HAPPY_ABOUT -> Bob's promotion
   - GOOD: "Nate plays games on a Gamecube" -> Nate -> PLAYS_GAMES_ON -> Gamecube (when "Gamecube" is in ENTITIES)
   - GOOD: "Alice congratulated Bob" (relationship between two entities), "Alice lives in Paris" (relationship between entity and place)
4. Do not emit semantically redundant facts, even across episodes within the CURRENT EPISODE.
However, if a later episode adds specific details to a previously stated fact (e.g., adding a brand
name, a count, a color, a location, or any concrete attribute), extract the more detailed version
as a NEW fact - it is NOT a duplicate. Only treat facts as duplicates when they convey the same
specificity.
   - NOT a duplicate: "user plays video games" (Episode 0)
   vs. "user plays games on a Gamecube" (Episode 1) -> extract the second, more detailed fact.
   - IS a duplicate: "user plays games on a Gamecube" (Episode 0)
   vs. "user plays Gamecube games" (Episode 1) -> extract once.
5. The 'fact' MUST preserve all specific details from the source text: proper nouns, brand names,
product names, model numbers, quantities, counts, colors, materials, physical descriptions,
specific items, named locations, and named activities. Paraphrase the sentence structure but NEVER
generalize:
   - NEVER generalize "Gamecube" to "gaming console", "Ford Mustang" to "car", "wool coat" to
"coat", "red and purple lighting" to "lighting", "cracked windshield" to "car damage", or "three
screenplays" to "several screenplays".
   - Do not verbatim quote the original text, but every concrete noun, number, and descriptor in
the source should survive into the 'fact'.
6. Facts should include entity names rather than pronouns whenever possible.

RELATION TYPE RULES:

- If FACT TYPES are provided and the relationship matches one of the types (considering the entity
type signature), use that factTypeName as the 'relationType'.
- Otherwise, derive a 'relationType' from the relationship predicate in SCREAMING_SNAKE_CASE
(e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH). Prefer a short, common predicate over a novel or compound one;
reach for the verb most people would reach for first.`;

function formatEntitiesBlock(nodes: ReadonlyArray<EntityNode>): string {
  if (nodes.length === 0) return 'None';
  return nodes
    .map((n, idx) => `- id: ${idx}, name: "${n.name}", labels: [${n.labels.join(', ')}]`)
    .join('\n');
}

export function buildExtractEdgesMessages(ctx: {
  episode: EpisodicNode;
  nodes: EntityNode[];
  previousEpisodes: EpisodicNode[];
  customInstructions?: string;
  edgeTypes?: EdgeTypeMap;
  edgeTypeMappings?: EdgeTypeMappings;
}): BaseMessage[] {
  const {
    episode,
    nodes,
    previousEpisodes,
    customInstructions,
    edgeTypes,
    edgeTypeMappings,
  } = ctx;

  const edgeTypeSignaturesMap: Record<string, string[]> = {};
  if (edgeTypeMappings) {
    for (const [[src, tgt], names] of edgeTypeMappings) {
      const sig = edgeTypeKey(src, tgt);
      for (const n of names) {
        (edgeTypeSignaturesMap[n] ??= []).push(sig);
      }
    }
  }

  const edgeTypesContext = edgeTypes
    ? Object.entries(edgeTypes).map(([name, { description }]) => ({
        factTypeName: name,
        factTypeSignatures: edgeTypeSignaturesMap[name] ?? ['Entity,Entity'],
        factTypeDescription: description,
      }))
    : [];

  let humanContent = `Apply every rule from the system instructions when extracting facts from the CURRENT EPISODE below.

<PREVIOUS EPISODES>
${formatPreviousEpisodes(previousEpisodes)}
</PREVIOUS EPISODES>

<CURRENT EPISODE>
${formatCurrentEpisode(episode)}
</CURRENT EPISODE>

<ENTITIES>
${formatEntitiesBlock(nodes)}
</ENTITIES>`;

  if (edgeTypesContext.length > 0) {
    humanContent += `

<FACT TYPES>
${JSON.stringify(edgeTypesContext, null, 2)}
</FACT TYPES>`;
  }

  if (customInstructions) {
    humanContent += `\n\n<CUSTOM INSTRUCTIONS>\n${customInstructions}\n</CUSTOM INSTRUCTIONS>`;
  }

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}

export function buildExtractEdgesValidator(ctx: {
  nodes: ReadonlyArray<unknown>;
}): (parsed: ExtractedEdgesOutput) => Violation[] {
  const nodeCount = ctx.nodes.length;

  return (parsed) => {
    const violations: Violation[] = [];
    for (const e of parsed.edges) {
      if (e.sourceEntityIdx >= nodeCount) {
        violations.push({
          code: 'edge.source-idx-out-of-range',
          message: `sourceEntityIdx ${e.sourceEntityIdx} is out of range (ENTITIES has ${nodeCount})`,
        });
      }
      if (e.targetEntityIdx >= nodeCount) {
        violations.push({
          code: 'edge.target-idx-out-of-range',
          message: `targetEntityIdx ${e.targetEntityIdx} is out of range (ENTITIES has ${nodeCount})`,
        });
      }
      if (e.sourceEntityIdx === e.targetEntityIdx) {
        violations.push({
          code: 'edge.self-loop',
          message: `self-loop: sourceEntityIdx and targetEntityIdx both refer to id ${e.sourceEntityIdx}`,
        });
      }
    }
    return violations;
  };
}
