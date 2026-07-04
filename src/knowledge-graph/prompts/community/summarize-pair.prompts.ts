import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { MAX_SUMMARY_CHARS } from '../text-utils';

// Schema

export const SummarySchema = z.object({
  summary: z
    .string()
    .max(MAX_SUMMARY_CHARS)
    .describe('Combined summary covering the durable facts from both inputs'),
});

export type SummaryOutput = z.infer<typeof SummarySchema>;

// Prompt builder

const SYSTEM_PROMPT = `You merge two factual summaries into a single dense factual summary.

Primary goal:
Synthesize the durable facts from BOTH inputs into one information-dense summary that stands alone without
reference to either source.

Why this matters:
The two inputs are themselves derived summaries, and your output replaces them as the durable record - it is
fed back as an input to later merges and the originals are NOT revisited. A supported fact you drop is
forgotten permanently; a fact you add that neither input supports compounds forward as if it were true. Errors
in either direction are self-reinforcing, so be exhaustive within the two inputs and NEVER go beyond them.

Rules:
- Use ONLY facts explicitly stated in the two SUMMARIES. NEVER infer, generalize, or add anything beyond what
they directly support.
- Preserve every materially relevant name, role, place, date, count, relationship, and change over time that
either input supports. Prefer retaining a supported concrete detail over omitting it for brevity.
- Merge overlapping facts rather than repeating them. When the two inputs conflict and one is clearly more
recent or more specific, prefer that one.
- Prefer compact factual sentences over vague thematic phrasing.
- When the durable fact is the content of what was said, state the content directly instead of narrating that
it was said. Use communication verbs only when the act of speaking, asking, sharing, presenting, or announcing
is itself the important fact.
- NEVER use meta-language verbs like "mentioned", "described", "stated", "noted", "discussed", "referenced",
"indicated", or "reported" unless the communication act itself is the fact. State facts directly.
- NEVER refer to the summaries, the merge, or the synthesis process. The output must read as a standalone brief.
- Write in third person.
- The combined summary MUST be under ${MAX_SUMMARY_CHARS} characters.

<EXAMPLES>
<SUMMARIES>
[0] Priya runs weekly cycling rides through Forest Park on Saturday mornings and tracks Strava segments along the Lower Loop.
[1] Priya switched from a road bike to a gravel bike in March 2025 and shares routes with Sam via a shared Komoot account.
</SUMMARIES>
GOOD output:
{"summary": "Priya runs weekly cycling rides through Forest Park on Saturday mornings and tracks Strava
segments along the Lower Loop. She switched from a road bike to a gravel bike in March 2025 and shares routes
with Sam via a shared Komoot account."}
BAD output:
{"summary": "Priya is an avid cyclist who is passionate about the outdoors. Summary [0] discusses her Saturday
rides and summary [1] mentions a bike upgrade and route sharing."}
</EXAMPLES>`;

function formatSummaries(summaries: readonly string[]): string {
  return summaries.map((s, i) => `[${i}] ${s}`).join('\n');
}

export function buildSummarizePairMessages(ctx: {
  summaries: [string, string];
}): BaseMessage[] {
  const { summaries } = ctx;

  const humanContent = `Apply every rule from the system instructions when synthesizing the summaries below.

<SUMMARIES>
${formatSummaries(summaries)}
</SUMMARIES>`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}
