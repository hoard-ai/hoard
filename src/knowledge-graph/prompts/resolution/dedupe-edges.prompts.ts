import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { EpisodicNode } from '@/knowledge-graph/models';
import type { Violation } from '@/llm';

import {
  formatCurrentEpisode,
  formatPreviousEpisodes,
  formatPromptTimestamp,
} from '../text-utils';

// Schema

export const EdgeDedupeSchema = z.object({
  duplicateFacts: z
    .array(z.int().nonnegative())
    .describe(
      'List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',
    ),
  contradictedFacts: z
    .array(z.int().nonnegative())
    .describe(
      'List of idx values of contradicted facts (from full idx range). Empty list if none.',
    ),
});

export type EdgeDedupeOutput = z.infer<typeof EdgeDedupeSchema>;

// Prompt builder

const SYSTEM_PROMPT = `You are an expert knowledge graph edge deduplication system.

Your task is to determine whether a newly extracted fact (edge) duplicates or contradicts existing facts.
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicateFacts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradictedFacts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

How the two lists were built:
- EXISTING FACTS were retrieved structurally: they connect the SAME two entities as the NEW FACT. Same
  endpoints is why they are here, so they are the only place a true duplicate can live.
- FACT INVALIDATION CANDIDATES were retrieved by embedding and keyword similarity to the NEW FACT. They are
  here BECAUSE they are on a close topic, NOT because they conflict - most will neither duplicate nor
  contradict. Topical closeness is the reason to inspect them, never evidence on its own. Flag one only when
  the NEW FACT genuinely makes it false.

1. DUPLICATE DETECTION:
- If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicateFacts.
- If no duplicates, return an empty list for duplicateFacts.

2. CONTRADICTION DETECTION:
- Determine which facts the NEW FACT contradicts from either list.
- A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
- Return all contradicted idx values in contradictedFacts.
- If no contradictions, return an empty list for contradictedFacts.

<EXAMPLES>
<NEW FACT>
- name: WORKS_AT, fact: "Alice joined Acme Corp in 2020"
</NEW FACT>
<EXISTING FACTS>
- idx: 0, name: WORKS_AT, fact: "Alice joined Acme Corp in 2020"
</EXISTING FACTS>
<FACT INVALIDATION CANDIDATES>
None
</FACT INVALIDATION CANDIDATES>
Result: {"duplicateFacts": [0], "contradictedFacts": []}
(identical factual information)

<NEW FACT>
- name: WORKS_AT, fact: "Alice works at Acme Corp as a senior engineer"
</NEW FACT>
<EXISTING FACTS>
- idx: 0, name: WORKS_AT, fact: "Alice works at Acme Corp as a software engineer"
</EXISTING FACTS>
<FACT INVALIDATION CANDIDATES>
None
</FACT INVALIDATION CANDIDATES>
Result: {"duplicateFacts": [], "contradictedFacts": [0]}
(same relationship but updated title - contradiction, NOT a duplicate)

<NEW FACT>
- name: RAN, fact: "Bob ran 3 miles on Wednesday"
</NEW FACT>
<EXISTING FACTS>
- idx: 0, name: RAN, fact: "Bob ran 5 miles on Tuesday"
</EXISTING FACTS>
<FACT INVALIDATION CANDIDATES>
None
</FACT INVALIDATION CANDIDATES>
Result: {"duplicateFacts": [], "contradictedFacts": []}
(different events on different days - neither duplicate nor contradiction)

<NEW FACT>
- name: WORKS_AT, fact: "Alice left Acme Corp in January 2024"
</NEW FACT>
<EXISTING FACTS>
- idx: 0, name: WORKS_AT, fact: "Alice works at Acme Corp"
</EXISTING FACTS>
<FACT INVALIDATION CANDIDATES>
- idx: 1, name: EMPLOYED_BY, fact: "Alice is employed by Acme Corp full-time"
</FACT INVALIDATION CANDIDATES>
Result: {"duplicateFacts": [], "contradictedFacts": [0, 1]}
(continuous indexing across both lists - the new fact contradicts the existing fact AND the similar-topic invalidation candidate)
</EXAMPLES>
`;

function formatEdges(edges: Array<{ idx: number; name: string; fact: string }>): string {
  if (edges.length === 0) return 'None';
  return edges
    .map((e) => `- idx: ${e.idx}, name: ${e.name}, fact: "${e.fact}"`)
    .join('\n');
}

function rangeLabel(offset: number, count: number): string {
  if (count === 0) return `index ${offset} (none)`;
  return count === 1 ? `index ${offset}` : `indices ${offset}-${offset + count - 1}`;
}

export type DedupeEdgesCtx = {
  episode: EpisodicNode;
  previousEpisodes: EpisodicNode[];
  newEdge: { name: string; fact: string };
  endpointEdges: Array<{ idx: number; name: string; fact: string }>;
  similarEdges: Array<{ idx: number; name: string; fact: string }>;
  referenceTime: Date;
  customInstructions?: string;
};

export function buildDedupeEdgesMessages(ctx: DedupeEdgesCtx): BaseMessage[] {
  const {
    episode,
    previousEpisodes,
    newEdge,
    endpointEdges,
    similarEdges,
    referenceTime,
    customInstructions,
  } = ctx;

  const previousEpisodesText = formatPreviousEpisodes(previousEpisodes);
  const endpointRange = rangeLabel(0, endpointEdges.length);
  const similarRange = rangeLabel(endpointEdges.length, similarEdges.length);

  let humanContent = `Apply every rule from the system instructions when deduplicating the NEW FACT against the existing facts below.

<REFERENCE TIME>
${formatPromptTimestamp(referenceTime)}
</REFERENCE TIME>

<PREVIOUS EPISODES>
${previousEpisodesText}
</PREVIOUS EPISODES>

<CURRENT EPISODE>
${formatCurrentEpisode(episode)}
</CURRENT EPISODE>

<NEW FACT>
- name: ${newEdge.name}, fact: "${newEdge.fact}"
</NEW FACT>

<EXISTING FACTS>
Same source -> target as new fact, ${endpointRange}
${formatEdges(endpointEdges)}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
Similar topic, ${similarRange}
${formatEdges(similarEdges)}
</FACT INVALIDATION CANDIDATES>`;

  if (customInstructions) {
    humanContent += `\n\n<CUSTOM INSTRUCTIONS>\n${customInstructions}\n</CUSTOM INSTRUCTIONS>`;
  }

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}

export function buildDedupeEdgesValidator(
  ctx: Pick<DedupeEdgesCtx, 'endpointEdges' | 'similarEdges'>,
): (parsed: EdgeDedupeOutput) => Violation[] {
  const endpointCount = ctx.endpointEdges.length;
  const totalCount = endpointCount + ctx.similarEdges.length;

  return (parsed) => {
    const violations: Violation[] = [];

    for (const idx of parsed.duplicateFacts) {
      if (idx < 0 || idx >= endpointCount) {
        violations.push({
          code: 'dedupe-edges.duplicate-idx-out-of-range',
          message: `duplicateFacts idx ${idx} must be in EXISTING FACTS range [0, ${endpointCount})`,
        });
      }
    }
    for (const idx of parsed.contradictedFacts) {
      if (idx < 0 || idx >= totalCount) {
        violations.push({
          code: 'dedupe-edges.contradicted-idx-out-of-range',
          message: `contradictedFacts idx ${idx} must be in full range [0, ${totalCount})`,
        });
      }
    }

    return violations;
  };
}
