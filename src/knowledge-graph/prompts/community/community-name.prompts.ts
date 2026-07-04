import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { NodeNameSchema } from '@/knowledge-graph/types';

// Schema

export const CommunityNameSchema = z.object({
  name: NodeNameSchema.describe(
    'Short noun-phrase label (2-6 words) identifying the community',
  ),
});

export type CommunityNameOutput = z.infer<typeof CommunityNameSchema>;

// Prompt builder

const SYSTEM_PROMPT = `You label clusters of related entities produced by automatic community detection on a knowledge graph.

The cluster you are naming is the output of an automatic Louvain partition over a directed graph of entities 
and their relationships. It is NOT a human-curated topic and NOT necessarily a clean encyclopedic category. 

Your task is to produce a short noun-phrase label that captures what binds the members together based on the cluster summary.

Use <CLUSTER SUMMARY> as the authoritative theme; mine <SECTIONS> for concrete proper nouns
(people, places, projects, named topics) the synthesis may have abstracted away.

Rules:
- Return a single noun phrase, 2 to 6 words. NEVER write a sentence.
- Lead with concrete distinguishing tokens (proper nouns, place, named topic, named org, named person) when the summary supports them. 
Avoid bare generic headers like "Health", "Family", "Work", "Hobbies".
- When EXISTING COMMUNITY NAMES are provided, pick a label that is NOT in that list and is distinguishable from every entry in it. 
Lean on a different distinguishing token rather than appending suffixes.
- NEVER include quotes, trailing punctuation, leading articles ("The ", "A ", "An "), or meta-language like "Community of", "Group of", "Cluster of".
- NEVER mention the summarization process, the graph, nodes, edges, clusters, or community detection in the label itself.
- Use Title Case for proper nouns; lowercase common words.

<EXAMPLE>
<CLUSTER SUMMARY>
Priya and Sam run weekly cycling rides through Forest Park on Saturday mornings; their group also tracks Strava segments
along the Lower Loop and shares routes via a shared Komoot account.
</CLUSTER SUMMARY>
<EXISTING COMMUNITY NAMES>
- Belmont Arts Center ceramics
- Denver tech meetup
</EXISTING COMMUNITY NAMES>
GOOD: {"name": "Forest Park Saturday cycling"}
BAD: {"name": "Cycling"} (generic; no distinguishing token)
</EXAMPLE>`;

export function buildCommunityNameMessages(ctx: {
  summary: string;
  existingNames: readonly string[];
  sections: readonly [string, string];
}): BaseMessage[] {
  const { summary, existingNames, sections } = ctx;

  const existingText =
    existingNames.length === 0 ? 'None' : existingNames.map((n) => `- ${n}`).join('\n');

  const humanContent = `Apply every rule from the system instructions when labeling the cluster below.

<CLUSTER SUMMARY>
${summary}
</CLUSTER SUMMARY>

<SECTIONS>
[A] ${sections[0]}
[B] ${sections[1]}
</SECTIONS>

<EXISTING COMMUNITY NAMES>
${existingText}
</EXISTING COMMUNITY NAMES>`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}
