import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { EpisodicNode } from '@/knowledge-graph/models';

import {
  concatenateEpisodes,
  MAX_SUMMARY_CHARS,
  truncateAtSentence,
} from '../text-utils';

// Schema

export const SagaSummarySchema = z.object({
  summary: z
    .string()
    .max(MAX_SUMMARY_CHARS)
    .describe('Factual knowledge brief for the saga'),
});

// Prompt builder

const SYSTEM_PROMPT = `You extract durable knowledge from message threads into a factual knowledge brief
about a SAGA (a named topic, project, or thread). The brief captures facts, decisions, preferences,
plans, entities, and relationships.

Primary goal:
Write a dense factual knowledge brief that captures all durable knowledge about the saga and stands alone
without reference to the original messages or conversation.

Capture explicitly stated:
- Facts and concrete details (names, dates, numbers, locations)
- Decisions and their outcomes
- Preferences and requirements (when a person explicitly claims them, attributed to that person)
- Plans, next steps, and commitments
- Relationships between entities (who works where, who owns what)
- State changes (what was X, now is Y)

Rules:
- Use ONLY facts explicitly stated in NEW EPISODES and durable facts already present in EXISTING SUMMARY.
NEVER infer beyond what is directly supported.
- NEVER use meta-language verbs: "mentioned", "described", "stated", "noted", "discussed", "referenced",
"indicated", "reported", "talked about", "brought up" - these describe conversational dynamics, not knowledge.
State facts directly instead.
- NEVER refer to the messages, conversation, thread, or participants' communicative acts. The output
must read as if no conversation happened - only the facts matter.
- NEVER begin with "This conversation", "The thread", "In this thread" or "The discussion".
- NEVER infer preferences or habits from a single passing mention. Only capture a preference when a
person explicitly states it ("I prefer X", "I love X", "I always do X") and attribute it to that person.
- Be exhaustive within the evidence. Preserve all names, dates, counts, and temporal qualifiers.
- Lead with the most important fact or decision.
- EXISTING SUMMARY contains previously extracted facts about the saga; merge new facts from NEW EPISODES
into it. When newer episodes contradict older facts, prefer the newer fact.
- If NEW EPISODES add no durable fact, return the existing summary unchanged.
- Write 2-6 dense sentences in third person.

<EXAMPLES>
Input:
<SAGA NAME>
Q1 Deployment Planning
</SAGA NAME>

<EXISTING SUMMARY>
None
</EXISTING SUMMARY>

<NEW EPISODES>
[Episode 0] (2025-02-20T14:00:00Z)
Jordan: We decided to move the deployment to March 15 instead of March 8. The staging environment isn't ready.
Priya: Agreed. I'll update the client timeline. We also need to switch from PostgreSQL to CockroachDB for the multi-region requirement.
</NEW EPISODES>

GOOD output:
{"summary": "Deployment moved from March 8 to March 15 because the staging environment is not ready.
Priya owns updating the client timeline. The database is switching from PostgreSQL to CockroachDB
to support the multi-region requirement."}

BAD output:
{"summary": "Jordan mentioned moving the deployment date. Priya discussed updating the timeline and
talked about switching databases. The team noted staging issues."}

Input:
<SAGA NAME>
Restaurant Recommendations
</SAGA NAME>

<EXISTING SUMMARY>
None
</EXISTING SUMMARY>

<NEW EPISODES>
[Episode 0] (2025-03-12T19:30:00Z)
Alex: I tried the new Thai place on Elm Street last night - the pad see ew was incredible. Definitely going back.
Mina: Oh nice, I've been wanting to try that. Is it the one next to the bookstore?
Alex: Yeah, Siam Kitchen. They're open until 11 PM on weekends.
</NEW EPISODES>

GOOD output:
{"summary": "Siam Kitchen is a Thai restaurant on Elm Street, next to a bookstore, open until 11 PM
on weekends. Alex considers the pad see ew excellent."}

BAD output:
{"summary": "Alex mentioned trying a new Thai place and discussed the pad see ew. Mina asked about
the location. Alex noted it was Siam Kitchen and stated the weekend hours."}

Input:
<SAGA NAME>
Team Working Preferences
</SAGA NAME>

<EXISTING SUMMARY>
None
</EXISTING SUMMARY>

<NEW EPISODES>
[Episode 0] (2025-04-02T10:15:00Z)
Sam: I really prefer working in the mornings - I'm way more productive before noon.
Dana: Same. I've been blocking 9-11 AM for deep work. Also, I can't stand Jira - can we move the tracker to Linear?
Sam: Fine by me. I'll set up the workspace.
</NEW EPISODES>

GOOD output:
{"summary": "Sam prefers morning work and reports higher productivity before noon. Dana blocks 9-11 AM
for deep work. Dana prefers Linear over Jira for issue tracking. Sam is setting up the Linear workspace."}

BAD output:
{"summary": "Sam and Dana discussed their work preferences. They talked about morning productivity and
mentioned switching from Jira to Linear."}
</EXAMPLES>`;

export function buildSummarizeSagasMessages(ctx: {
  sagaName: string;
  existingSummary: string;
  newEpisodes: EpisodicNode[];
}): BaseMessage[] {
  const { sagaName, existingSummary, newEpisodes } = ctx;

  const episodesText = concatenateEpisodes(newEpisodes);

  const existingText = existingSummary.trim()
    ? truncateAtSentence(existingSummary, MAX_SUMMARY_CHARS)
    : 'None';

  const humanContent = `Apply every rule from the system instructions when producing the knowledge brief.

<SAGA NAME>
${sagaName}
</SAGA NAME>

<EXISTING SUMMARY>
${existingText}
</EXISTING SUMMARY>

<NEW EPISODES>
${episodesText}
</NEW EPISODES>`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}
