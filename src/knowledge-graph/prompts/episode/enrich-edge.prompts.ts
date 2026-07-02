import { BaseMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { EpisodicNode } from '@/knowledge-graph/models';
import type { Violation } from '@/llm';

import { formatCurrentEpisode, formatPromptTimestamp } from '../text-utils';

// Schema
//
// The unified edge-enrichment call extracts temporal bounds for EVERY surviving
// edge and, when the edge has a custom fact type, its typed attributes in the
// same call. Custom attributes are nested under `attributes` rather than merged
// flat so a custom field literally named validAt/invalidAt cannot collide with
// the edge's own temporal bounds.

const EdgeTemporalSchema = z.object({
  validAt: z.iso
    .datetime()
    .nullable()
    .optional()
    .describe(
      'When the fact became true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z). Null if no temporal information.',
    ),
  invalidAt: z.iso
    .datetime()
    .nullable()
    .optional()
    .describe(
      'When the fact stopped being true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z). Null if ongoing or unknown.',
    ),
});

export type EdgeEnrichmentOutput = z.infer<typeof EdgeTemporalSchema> & {
  attributes?: Record<string, unknown>;
};

/**
 * Builds the enrichment response schema. Temporal bounds always sit at the top
 * level; a custom fact-type schema (a non-empty z.object) is nested under
 * `attributes`. Untyped edges get the temporal schema alone. The cast lets the
 * caller infer `EdgeEnrichmentOutput` through invokeStructured's generic (the
 * runtime schema is a real Zod object either way).
 */
export function buildEnrichEdgeSchema(
  customSchema?: z.ZodType,
): z.ZodType<EdgeEnrichmentOutput> {
  const schema = customSchema
    ? EdgeTemporalSchema.extend({ attributes: customSchema })
    : EdgeTemporalSchema;
  return schema as z.ZodType<EdgeEnrichmentOutput>;
}

// Prompt builder

export const EDGE_ENRICHMENT_SYSTEM_PROMPT = `You enrich a single FACT (a knowledge-graph edge) using the FACT text and the
CURRENT EPISODE excerpt it came from. You extract the fact's temporal bounds and, when the response
schema includes an "attributes" object, its typed attribute values. You output strictly the JSON
specified by the response schema - no reasoning, no explanation, no commentary in any field.

TEMPORAL RULES:
- Determine when the fact became true (validAt) and when it stopped being true (invalidAt).
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set validAt to REFERENCE TIME.
- If a change or end is expressed, set invalidAt to the relevant time.
- Leave a bound null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
- Use ISO 8601 with "Z" suffix (UTC), e.g. 2025-04-30T00:00:00Z.
- NEVER hallucinate or infer dates from unrelated events.

ATTRIBUTE RULES (apply only when the response schema includes an "attributes" object):
HARD RULES - violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the FACT or CURRENT EPISODE,
   (b) the existing value already in EXISTING ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description tells you the FORMAT a real value should take. The
   description text is NEVER itself a value. NEVER copy schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the provided context, set the field to null (or omit it) - do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes. If you cannot produce a clean value of that type from the provided context,
   the field is null.

6. Preserve existing attribute values unless the FACT or CURRENT EPISODE explicitly provides
   a new value.`;

const WEEKDAYS = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
] as const;
const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const;

function daysInMonth(year: number, month: number): number {
  return new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
}

function isoDate(year: number, month: number, day: number): string {
  return formatPromptTimestamp(new Date(Date.UTC(year, month, day)));
}

function addDays(base: Date, days: number): string {
  return isoDate(base.getUTCFullYear(), base.getUTCMonth(), base.getUTCDate() + days);
}

function addMonthsClamped(
  year: number,
  month: number,
  day: number,
  delta: number,
): string {
  const target = month + delta;
  const ty = year + Math.floor(target / 12);
  const tm = ((target % 12) + 12) % 12;
  return isoDate(ty, tm, Math.min(day, daysInMonth(ty, tm)));
}

/**
 * A block of exact ISO 8601 anchors and calendar facts derived from
 * `referenceTime`, so the model looks up dates it would otherwise miscompute
 * (month-end clamps, leap years, weekdays, quarters).
 */
export function buildTimeReferenceTable(referenceTime: Date): string {
  // TODO(timezone): computed in UTC; resolve against the user's timezone once
  // per-user timezones exist (weekday, day-of-month, day boundaries all shift).
  const y = referenceTime.getUTCFullYear();
  const m = referenceTime.getUTCMonth();
  const d = referenceTime.getUTCDate();
  const monthLen = daysInMonth(y, m);
  const febLen = daysInMonth(y, 1);
  const q = Math.floor(m / 3);
  const toMonday = referenceTime.getUTCDay() === 0 ? 6 : referenceTime.getUTCDay() - 1;

  const facts = [
    `- today is a ${WEEKDAYS[referenceTime.getUTCDay()]}`,
    `- this month (${MONTHS[m]}) has ${monthLen} days`,
    `- February this year has ${febLen} days${febLen === 29 ? ' (leap year)' : ''}`,
    `- current quarter: Q${q + 1} (${isoDate(y, q * 3, 1)} to ${isoDate(y, q * 3 + 2, daysInMonth(y, q * 3 + 2))})`,
  ];
  const dates = [
    `- yesterday: ${addDays(referenceTime, -1)}`,
    `- tomorrow: ${addDays(referenceTime, 1)}`,
    `- 1 week ago: ${addDays(referenceTime, -7)}`,
    `- 2 weeks ago: ${addDays(referenceTime, -14)}`,
    `- 1 month ago: ${addMonthsClamped(y, m, d, -1)}`,
    `- 3 months ago: ${addMonthsClamped(y, m, d, -3)}`,
    `- 6 months ago: ${addMonthsClamped(y, m, d, -6)}`,
    `- start of this week (Monday): ${addDays(referenceTime, -toMonday)}`,
    `- end of this week (Sunday): ${addDays(referenceTime, 6 - toMonday)}`,
    `- start of this month: ${isoDate(y, m, 1)}`,
    `- end of this month: ${isoDate(y, m, monthLen)}`,
  ];

  return `<TIME REFERENCE>
Calendar facts and exact ISO 8601 values derived from REFERENCE TIME. Use these to resolve relative expressions instead of computing dates yourself; month offsets are clamped to the last valid day of the target month. Week starts Monday.

Facts:
${facts.join('\n')}

Resolved dates:
${dates.join('\n')}
</TIME REFERENCE>`;
}

export function buildEnrichEdgeMessages(ctx: {
  fact: string;
  episode: EpisodicNode;
  referenceTime: Date;
  existingAttributes?: Record<string, unknown>;
  hasCustomAttributes: boolean;
}): BaseMessage[] {
  const { fact, episode, referenceTime, existingAttributes, hasCustomAttributes } = ctx;

  let humanContent = `Apply every rule from the system instructions when enriching the fact below.

<FACT>
${fact}
</FACT>

<CURRENT EPISODE>
${formatCurrentEpisode(episode)}
</CURRENT EPISODE>

<REFERENCE TIME>
${formatPromptTimestamp(referenceTime)}
</REFERENCE TIME>

${buildTimeReferenceTable(referenceTime)}`;

  if (hasCustomAttributes) {
    humanContent += `

<EXISTING ATTRIBUTES>
${JSON.stringify(existingAttributes ?? {}, null, 2)}
</EXISTING ATTRIBUTES>`;
  }

  return [
    new SystemMessage(EDGE_ENRICHMENT_SYSTEM_PROMPT),
    new HumanMessage(humanContent),
  ];
}

// Both bounds are schema-validated ISO 8601, so Date.parse is safe here.
function timestampOrderViolation(
  validAt: string | null | undefined,
  invalidAt: string | null | undefined,
): Violation | null {
  if (!validAt || !invalidAt) return null;
  return Date.parse(validAt) > Date.parse(invalidAt)
    ? {
        code: 'edge.invalid-temporal-order',
        message: `invalidAt (${invalidAt}) must not precede validAt (${validAt})`,
      }
    : null;
}

export function buildEnrichEdgeValidator(): (
  parsed: EdgeEnrichmentOutput,
) => Violation[] {
  return (parsed) => {
    const v = timestampOrderViolation(parsed.validAt, parsed.invalidAt);
    return v ? [v] : [];
  };
}
