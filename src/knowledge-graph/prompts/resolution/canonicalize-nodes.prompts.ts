import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { Violation } from '@/llm';

// Schema

export const CanonicalizeNodesSchema = z.object({
  duplicateGroups: z
    .array(z.array(z.int().nonnegative()).min(2))
    .describe(
      'Groups of entity ids that denote the same real-world entity. The FIRST id in each group is the canonical entity (the most complete, descriptive name). Omit entities that have no duplicates; empty array when there are none.',
    ),
});

export type CanonicalizeNodesOutput = z.infer<typeof CanonicalizeNodesSchema>;

// Prompt builder

const SYSTEM_PROMPT = `You are an expert knowledge graph deduplication system.

The ENTITIES below were extracted from partial, overlapping views (chunks) of ONE document by
extractors that could not see the other chunks. As a result, the same real-world entity may appear
several times under different names: a full name in one chunk, a short form ("Tomas") or a role
description ("the captain", "biologist") in another.

Your task is to group the ids of entries that denote the SAME real-world entity.

Evidence:
Each entry carries an identifyingDescription (a standalone descriptor of who or what the entity is) and
aliases (the surface forms its chunk used for it). Two entries denote the same entity when their
identifyingDescriptions describe the same referent, or when one entry's name appears among another's aliases
AND the identifyingDescriptions agree on what kind of thing it is.
A shared alias or epithet alone is NOT evidence of identity: narrative text often applies the
same epithet ("the rival", "the visitor") to a person AND to their vessel, organization, or
creation. If the identifyingDescriptions describe different kinds of referent (a person vs a ship, a scientist
vs an institution), the entries are distinct no matter what surface forms they share.

Observed pronouns:
Each entry also carries referredToAsPronouns - the pronouns its own chunk's text used for it.
These are observations from the text, not inferences. CONFLICTING pronouns ("he" vs "she", or a
person pronoun vs "it") are strong evidence of DISTINCT referents. Matching pronouns are only weak
support - many entities share a pronoun, so agreement alone NEVER establishes identity. An empty
list means no pronoun was observed, not that none applies.

Cost of error is asymmetric:
Grouping permanently collapses nodes into one and repoints every edge between them. That is far
harder to undo than leaving a genuine duplicate unmerged, which a later pass can still catch. When
the evidence does not clearly establish the SAME referent, do NOT group. Related is not the same:
a person and their employer, a person and their vessel, two colleagues, an object and its owner,
two holders of similar roles are all distinct.

Output rules:
1. Within each group, put the CANONICAL entry FIRST: the one whose name is the most complete and
descriptive (prefer "Dr. Elena Marquez" over "biologist", "Tomas Feng" over "Tomas").
2. Only output groups of two or more ids. Entities with no duplicate are omitted entirely.
3. An id may appear in at most one group.

<EXAMPLES>
<ENTITIES>
- id: 0, name: "Dr. Elena Marquez", labels: [Entity, Person], identifyingDescription: "marine biologist at the University of Concepcion leading the expedition", aliases: ["Elena", "Dr. Marquez"], referredToAsPronouns: ["she"]
- id: 1, name: "Meridian", labels: [Entity], identifyingDescription: "converted trawler serving as the expedition's research vessel", aliases: ["the trawler"], referredToAsPronouns: ["she"]
- id: 2, name: "biologist", labels: [Entity, Person], identifyingDescription: "marine biologist aboard the vessel securing the specimen tanks", aliases: ["the biologist"], referredToAsPronouns: ["she"]
- id: 3, name: "Captain Ruiz", labels: [Entity, Person], identifyingDescription: "captain of the research vessel", aliases: ["Ruiz"], referredToAsPronouns: []
- id: 4, name: "first officer", labels: [Entity, Person], identifyingDescription: "first officer keeping peace between the captain and the scientists", aliases: ["the first officer"], referredToAsPronouns: []
- id: 5, name: "Kaiyo", labels: [Entity], identifyingDescription: "rival research vessel that arrived at the survey site", aliases: ["the rival"], referredToAsPronouns: ["it"]
- id: 6, name: "Dr. Haruki Sato", labels: [Entity, Person], identifyingDescription: "scientist leading the rival research vessel", aliases: ["the rival"], referredToAsPronouns: ["he"]
</ENTITIES>
Result: {"duplicateGroups": [[0, 2]]}
(both identifyingDescriptions describe the expedition's marine biologist and both chunks refer to
her as "she" - same person, full name first; the first officer matches no other entry, and holding
a role aboard the same vessel is not evidence of identity with the captain; the Kaiyo and Dr. Sato
share the epithet "the rival" but one is a vessel the text calls "it" and the other a person the
text calls "he" - different kinds of referent, never grouped; the Meridian is also called "she",
and sharing a pronoun with Dr. Marquez is not evidence of identity - a vessel and a person are
different kinds of referent)
</EXAMPLES>`;

export type CanonicalizeNodesCtx = {
  entities: Array<{
    id: number;
    name: string;
    labels: readonly string[];
    identifyingDescription: string;
    aliases: string[];
    referredToAsPronouns: string[];
  }>;
};

export function buildCanonicalizeNodesMessages(ctx: CanonicalizeNodesCtx): BaseMessage[] {
  const entitiesText = ctx.entities
    .map((e) => {
      const aliases = e.aliases.map((a) => `"${a}"`).join(', ');
      const pronouns = e.referredToAsPronouns.map((p) => `"${p}"`).join(', ');
      return `- id: ${e.id}, name: "${e.name}", labels: [${e.labels.join(', ')}], identifyingDescription: "${e.identifyingDescription}", aliases: [${aliases}], referredToAsPronouns: [${pronouns}]`;
    })
    .join('\n');

  const n = ctx.entities.length;
  const humanContent = `Apply every rule from the system instructions when grouping the entities below.

<ENTITIES>
${entitiesText}
</ENTITIES>

ENTITIES contains ${n} entities with ids 0 through ${n - 1}. Group only ids that denote the same real-world entity; output an empty duplicateGroups array when there are none.`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}

export function buildCanonicalizeNodesValidator(ctx: {
  entities: ReadonlyArray<unknown>;
}): (parsed: CanonicalizeNodesOutput) => Violation[] {
  const count = ctx.entities.length;

  return (parsed) => {
    const violations: Violation[] = [];
    const seen = new Set<number>();
    for (const group of parsed.duplicateGroups) {
      for (const id of group) {
        if (id >= count) {
          violations.push({
            code: 'canonicalize-nodes.id-out-of-range',
            message: `id ${id} is out of range (ENTITIES has ${count})`,
          });
        }
        if (seen.has(id)) {
          violations.push({
            code: 'canonicalize-nodes.duplicate-id',
            message: `id ${id} appears in more than one group or twice in a group`,
          });
        }
        seen.add(id);
      }
    }
    return violations;
  };
}
