import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { EpisodicNode } from '@/knowledge-graph/models';
import { NodeNameSchema } from '@/knowledge-graph/types';
import type { Violation } from '@/llm';

import { buildCoreferenceBlock, type CorefChunkGroup } from '../coref-utils';
import {
  formatCurrentEpisode,
  formatPreviousEpisodes,
  MAX_SUMMARY_CHARS,
} from '../text-utils';

// Schema

export const NodeSummarySchema = z.object({
  summaries: z
    .array(
      z.object({
        name: NodeNameSchema.describe(
          'Name of the entity being summarized (echoed verbatim from the input)',
        ),
        summary: z
          .string()
          .max(MAX_SUMMARY_CHARS)
          .describe(`Updated summary for the entity`),
      }),
    )
    .describe(
      'List of entity summaries. Only include entities that need summary updates.',
    ),
});

export type NodeSummaryOutput = z.infer<typeof NodeSummarySchema>;

// Prompt builder

const SYSTEM_PROMPT = `You maintain detailed, information-dense entity memories from episode text.

Use ONLY facts explicitly stated in the PREVIOUS EPISODES and CURRENT EPISODE, and durable facts
already present in each entity's existingSummary field. NEVER infer beyond what is directly supported.

Primary goal:
Write a dense factual summary of the entity that preserves as many supported details as possible while staying coherent and durable.

Why this matters:
The summary you produce becomes the entity's durable memory and is fed back to you as existingSummary on the
next pass. The original episodes are NOT revisited. A supported detail you drop is forgotten permanently; an
unsupported detail you add compounds forward as if it were fact. Errors in either direction are self-reinforcing,
which is why you must be exhaustive within the evidence and NEVER go beyond it.

When the input includes ENTITY TYPE DESCRIPTIONS, use them to decide which facts are most relevant
to the entity type. NEVER mention the entity type, type description, or classification in the summary text itself.

What to capture:
- Stable facts about the entity
- All materially relevant named people, organizations, places, events, documents, objects, and other entities linked to it
- Explicit actions, roles, responsibilities, relationships, and outcomes
- Counts, sequences, and repeated patterns when the evidence supports them
- Temporal details at the highest fidelity available: dates, months, years, ordering, and changes over time
- Current state over superseded state when newer episodes clearly update older information

Rules:
- Be exhaustive within the evidence. Prefer retaining a supported concrete detail over omitting it for brevity.
- NEVER infer preferences, habits, recurrence, frequency, causality, intent, importance, or category
from a name, a single mention, or weak evidence.
- Only describe something as recurring, preferred, typical, habitual, or ongoing when multiple episodes
explicitly support that claim or one episode states it directly.
- Include all materially relevant named participants that appear in the evidence.
- Include temporal qualifiers whenever they are available.
- Mention counts when they are directly supported and meaningful. Prefer direct factual phrasing
over meta phrasing.
- When the durable fact is the content of what was said, state the content directly instead of
describing that it was said.
- Use communication verbs only when the act of speaking, asking, sharing, presenting,
announcing, or telling is itself the important fact.
- NEVER manufacture pattern language from a single occurrence. A single mention can support a fact,
but not a trend, habit, or preference unless the text states that directly.
- If the evidence is insufficient or ambiguous, omit the claim.
- NEVER mention the source material or summarization process.
- NEVER mention episodes, messages, prompts, summaries, memory, graphs, nodes, labels, node types,
ontology, schema, or categorization.
- NEVER output phrases like "the summary", "the entity", "categorized as", "tagged as", "suggests",
"implies", "appears to", or "recorded interaction".
- NEVER use "the entity" as a pronoun. Use the entity's actual name or a natural pronoun
(he, she, it, they).
- NEVER use meta-language verbs like "mentioned", "described", "stated", "noted", "discussed",
"referenced", "indicated", or "reported". State the fact directly instead of describing how it
was communicated.
- NEVER begin the summary with "A ", "An ", or "This is". If the entity's name starts with
"The" (e.g. "The Washington Post"), that is acceptable; otherwise NEVER lead with "The ".
Lead with the entity's name or a concrete fact.
- When newer episode text conflicts with older summary content, prefer the newer explicit fact.
- Omit an entity from the output when its summary does not need to change - either the new
episodes add no durable fact, or there is no existing summary and no relevant information in
the episodes. Omitted entities preserve their existing value (or absence) on the caller side.
- The summary should read like a compact brief, not a tagline.
- Write 2-6 dense sentences in third person.
- Return only the summary text.

<EXAMPLES>
Input:
<PREVIOUS EPISODES>
- [workshop-recap] (2025-03-03T00:00:00Z): Mina: Jordan Lee presented a ceramics workshop at Belmont Arts Center on March 3, 2025. 
The workshop had 24 attendees and focused on wheel-thrown bowls.
- [followup-announcement] (2025-03-10T00:00:00Z): Owen: After the session, Jordan announced a second April workshop for returning students.
</PREVIOUS EPISODES>

<CURRENT EPISODE>
Name: studio-update
Timestamp: 2025-04-14T00:00:00Z
Content: Mina: Jordan shared that the new kiln room opened last month and that Jordan now supervises two studio assistants. 
Owen: Jordan still teaches beginner ceramics on Wednesday evenings.
</CURRENT EPISODE>

<ENTITY TYPE DESCRIPTIONS>
- Person: A human being.
</ENTITY TYPE DESCRIPTIONS>

<ENTITIES>
- name: "Jordan Lee", type: "Person", existingSummary: "Jordan Lee works at Belmont Arts Center.",
facts: ["Jordan Lee presented a ceramics workshop at Belmont Arts Center",
"Jordan Lee announced a second April workshop", "Jordan Lee supervises two studio assistants",
"Jordan Lee teaches beginner ceramics on Wednesday evenings"]
</ENTITIES>

GOOD output:
{"summaries": [{"name": "Jordan Lee", "summary": "Jordan Lee works at Belmont Arts Center.
Jordan presented a ceramics workshop there on March 3, 2025 for 24 attendees focused on
wheel-thrown bowls, and later announced a second April workshop for returning students.
Jordan supervises two studio assistants, teaches beginner ceramics on Wednesday evenings,
and works out of the new kiln room that opened the previous month."}]}

BAD output:
{"summaries": [{"name": "Jordan Lee", "summary": "Jordan Lee seems interested in ceramics.
Jordan mentioned teaching and was described as busy at the arts center."}]}
</EXAMPLES>`;

export function buildNodeSummaryMessages(ctx: {
  episode: EpisodicNode;
  previousEpisodes: EpisodicNode[];
  nodes: Array<{
    name: string;
    type?: string;
    existingSummary: string;
    facts: string[];
  }>;
  entityTypeDescriptions?: Record<string, string>;
  coreferences?: CorefChunkGroup[];
  labelChunks?: boolean;
}): BaseMessage[] {
  const {
    episode,
    previousEpisodes,
    nodes,
    entityTypeDescriptions,
    coreferences,
    labelChunks,
  } = ctx;

  const previousEpisodesText = formatPreviousEpisodes(previousEpisodes);

  const entitiesText = nodes
    .map((n) => {
      const factsText = n.facts.length > 0 ? JSON.stringify(n.facts) : '[]';
      const typePart = n.type ? `, type: "${n.type}"` : '';
      return `- name: "${n.name}"${typePart}, existingSummary: "${n.existingSummary}", facts: ${factsText}`;
    })
    .join('\n');

  let humanContent = `Apply every rule from the system instructions when producing summaries for the entities below.

<PREVIOUS EPISODES>
${previousEpisodesText}
</PREVIOUS EPISODES>

<CURRENT EPISODE>
${formatCurrentEpisode(episode)}
</CURRENT EPISODE>`;

  if (entityTypeDescriptions && Object.keys(entityTypeDescriptions).length > 0) {
    const descriptionsText = Object.entries(entityTypeDescriptions)
      .map(([label, description]) => `- ${label}: ${description}`)
      .join('\n');
    humanContent += `\n\n<ENTITY TYPE DESCRIPTIONS>\n${descriptionsText}\n</ENTITY TYPE DESCRIPTIONS>`;
  }

  const corefBlock = coreferences?.length
    ? buildCoreferenceBlock({ committed: coreferences, labelChunks })
    : null;
  if (corefBlock) {
    humanContent += `\n\n${corefBlock}\n\nThe COREFERENCES above were resolved during extraction; each applies where its quoted occurrence appears in the text.`;
  }

  humanContent += `\n\n<ENTITIES>\n${entitiesText}\n</ENTITIES>`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}

export function buildNodeSummaryValidator(ctx: {
  nodes: ReadonlyArray<{ name: string }>;
}): (parsed: NodeSummaryOutput) => Violation[] {
  const validNames = new Set(ctx.nodes.map((n) => n.name));

  return (parsed) => {
    const violations: Violation[] = [];
    const seen = new Set<string>();

    for (const s of parsed.summaries) {
      if (!validNames.has(s.name)) {
        violations.push({
          code: 'summary.unknown-name',
          message: `name "${s.name}" is not in the input ENTITIES set`,
        });
      }
      if (seen.has(s.name)) {
        violations.push({
          code: 'summary.duplicate-name',
          message: `duplicate name "${s.name}" in summaries`,
        });
      }
      seen.add(s.name);
    }
    return violations;
  };
}
