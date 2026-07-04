import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

import { formatPromptTimestamp } from '../text-utils';

// Schema

// Response schema is dynamic per call: the caller derives it from the matched
// EntityTypeMap entry's `schema` field at invocation time (see
// EpisodeService.fillEntityAttributesImpl), so no static schema is bound here.

// Prompt builder

const SYSTEM_PROMPT = `You are an entity attribute extraction specialist. You ONLY emit attribute values
that are explicitly stated in the EPISODE (or PREVIOUS EPISODES / RELATED FACTS, when provided) or
already present in EXISTING ATTRIBUTES. You output strictly the JSON specified by the response
schema - no reasoning, no explanation, no commentary in any field.

HARD RULES - violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the EPISODE (or PREVIOUS EPISODES / RELATED FACTS),
   (b) the existing value already in EXISTING ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...",
     "if no value is present...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description (e.g. an "Industry sector" field whose description
   reads "Industry classification, single word where possible") tells you the FORMAT a
   real value should take. The description text is NEVER itself a value. NEVER copy
   schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the provided context, set the field to null (or omit it) - do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes (a phone number, an industry name, a URL, a postal address). If you cannot
   produce a clean value of that type from the provided context, the field is null.

6. NEVER infer attribute values from the ENTITY NAME, from related entities, from
   generic world knowledge, or from prior summaries. Only verbatim or directly normalized
   text from the EPISODE, PREVIOUS EPISODES, or RELATED FACTS qualifies as a new value.

7. If the provided context contains no information about an attribute, leave the existing
   value in EXISTING ATTRIBUTES unchanged. If there is no existing value, the field is null.

8. Use REFERENCE TIME to resolve any relative temporal expressions in the provided context.

<EXAMPLES>
<ENTITY NAME>
Sam Rivera
</ENTITY NAME>
<EXISTING ATTRIBUTES>
{"phones": "415-555-0142"}
</EXISTING ATTRIBUTES>
The EPISODE contains no phone information for Sam.
GOOD -> "phones": "415-555-0142"   (preserved existing value)
BAD  -> "phones": "415-555-0142 (implied by existing attributes, but no new information in
        episode, retaining original value as per instruction...)"

<ENTITY NAME>
Northwind
</ENTITY NAME>
<EXISTING ATTRIBUTES>
{"industry": null}
</EXISTING ATTRIBUTES>
The EPISODE mentions Northwind only as the platform some content was posted to.
GOOD -> "industry": null   (no explicit industry classification was stated)
BAD  -> "industry": "Content platform, SaaS (implied by usage context, though not stated
        explicitly as industry classification...)"

<ENTITY NAME>
Priya
</ENTITY NAME>
<EXISTING ATTRIBUTES>
{}
</EXISTING ATTRIBUTES>
The EPISODE contains no phone for Priya, but discusses a project she contributed to.
GOOD -> "phones": null
BAD  -> "phones": "Worked with Lin and Marco on the Q3 launch..."   (off-topic content dump)
</EXAMPLES>`;

export function buildFillEntityAttributesMessages(ctx: {
  entityName: string;
  episodeContent: string;
  previousEpisodesContent: string[];
  relatedFacts: string[];
  referenceTime: Date;
  existingAttributes: Record<string, unknown>;
}): BaseMessage[] {
  const {
    entityName,
    episodeContent,
    previousEpisodesContent,
    relatedFacts,
    referenceTime,
    existingAttributes,
  } = ctx;

  let humanContent = `Apply every rule from the system instructions when updating the attributes for the entity below.

<ENTITY NAME>
${entityName}
</ENTITY NAME>

<EPISODE>
${episodeContent}
</EPISODE>`;

  if (previousEpisodesContent.length > 0) {
    humanContent += `

<PREVIOUS EPISODES>
${previousEpisodesContent.join('\n---\n')}
</PREVIOUS EPISODES>`;
  }

  if (relatedFacts.length > 0) {
    humanContent += `

<RELATED FACTS>
${relatedFacts.join('\n')}
</RELATED FACTS>`;
  }

  humanContent += `

<REFERENCE TIME>
${formatPromptTimestamp(referenceTime)}
</REFERENCE TIME>

<EXISTING ATTRIBUTES>
${JSON.stringify(existingAttributes, null, 2)}
</EXISTING ATTRIBUTES>`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}
