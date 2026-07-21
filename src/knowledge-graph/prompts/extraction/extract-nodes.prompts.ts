import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { EntityTypeMap } from '@/knowledge-graph/episode/types';
import {
  EntityCorefDescriptorSchema,
  UnresolvedReferenceSchema,
} from '@/knowledge-graph/extraction/types';
import type { EpisodicNode } from '@/knowledge-graph/models';
import { EpisodeType } from '@/knowledge-graph/models';
import { NodeNameSchema } from '@/knowledge-graph/types';
import type { Violation } from '@/llm';

import { formatCurrentEpisode, formatPreviousEpisodes } from '../text-utils';

// Schema

// TODO: Conditionally include `entityTypeId` in the schema and a corresponding classification
// note in the prompt only when ENTITY TYPES are provided. Otherwise the example labels
// "(Person)", "(Location)", etc. read as a fixed taxonomy, which they are not - the actual
// types are user-supplied and may not include any of those labels.
const ExtractedEntitySchema = z.object({
  name: NodeNameSchema.describe('Name of the extracted entity'),
  entityTypeId: z
    .int()
    .nonnegative()
    .optional()
    .describe(
      'ID of the classified entity type. Must be one of the integer IDs from the ENTITY TYPES list provided in the user message. Omit when no provided type fits.',
    ),
  // TODO: Multi-episode extraction per prompt
  // episodeIndices: z
  //   .array(z.number())
  //   .default([0])
  //   .describe(
  //     'List of episode numbers (0-indexed) this entity was extracted from. When processing a single episode, this should be [0].',
  //   ),
});

export const ExtractedEntitiesSchema = z.object({
  extractedEntities: z
    .array(ExtractedEntitySchema.extend(EntityCorefDescriptorSchema.shape))
    .describe('List of extracted entities'),
  unresolvedReferences: z
    .array(UnresolvedReferenceSchema)
    .default([])
    .describe(
      'Referring expressions that clearly point to a specific entity but could not be tied to any extracted entity. Empty when every reference resolves.',
    ),
});

export type ExtractedEntitiesOutput = z.infer<typeof ExtractedEntitiesSchema>;

// Prompt builders

const EXTRACTION_RULES = `
Why specificity matters:
Every node you extract becomes a standalone, independently-retrievable memory atom. It is stored on its own
and surfaced to the user weeks or months later with none of the surrounding sentence for context. So the test
behind every rule below is a single question: read entirely alone, later, would this name still identify one
specific, distinguishable thing? If it would be vague, generic, or ambiguous without its sentence, do NOT
extract it.

NEVER EXTRACT any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per episode,
  even if it appears multiple times or both as a speaker label and in the body text.
- Bare relational or kinship terms (dad, mom, mother, father, sister, brother, husband, wife,
  spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor,
  roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). These are too generic on
  their own. Instead, qualify them with the possessor: extract "Nisha's dad" not "dad",
  "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")
- Entities mentioned only in PREVIOUS EPISODES. That section is provided for context only -
  extract only entities explicitly present in the CURRENT EPISODE.
- Relationships or actions (these are extracted separately as edges).
- Temporal information like dates, times, or years.

Be explicit in node names, using full names and avoiding abbreviations.

Only extract entities specific enough to be uniquely identifiable. Ask: "Could this have its own
Wikipedia article or database entry, OR is it specific enough to distinguish from other items of
the same category within this scope?"

For objects, possessions, and physical items, extract when they are specific enough
to distinguish from other items of the same category. SHOULD be extracted:
- Brand-named items ("Gamecube", "Ford Mustang", "Moen faucet")
- Qualified items ("wool coat", "red and purple lighting", "cracked windshield",
  "dog leash")
- Items with a concrete distinguishing descriptor (color, material, size, model,
  owner, specific use)

Generic-claim exception: the bare-noun exclusions above are overridden when the bare
noun is the SUBJECT of a generic, declarative claim about the kind itself - i.e. the
sentence states a property that holds for the category as a whole, not for a specific
instance in this conversation. In that case, extract the bare noun as a node so the
claim can be captured. Examples:
- "Frogs are slow" -> extract "frogs" (claim about the kind)
- "Wolves hunt in packs" -> extract "wolves" (claim about the kind)
- "I saw three frogs" -> do NOT extract "frogs" (specific instances, not a generic claim)
- "His dog ran away" -> do NOT extract "dog" (specific instance - qualify as possessor's dog instead)

Specificity: always use the most specific form available in the source. If the source says
"road cycling", extract "road cycling" not "cycling". If it says "wool coat", extract "wool coat"
not "coat". When surrounding context makes an object's type clear, include that context in the
name (e.g., extract "dog leash" not "leash" when the passage discusses a dog walk). If a phrase
would not be meaningful and distinguishable when read alone later, do NOT extract it.

Use the descriptions in ENTITY TYPES to classify each extracted entity. Assign the appropriate 'entityTypeId' for each one.
If no entity type fits, omit entityTypeId.`;

const MESSAGE_SYSTEM_PROMPT = `You are an entity extraction specialist for conversational messages.
NEVER extract abstract concepts, feelings, or generic words.

Your task is to extract entity nodes that are explicitly mentioned in the CURRENT EPISODE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of
the referenced entities. Only extract distinct entities from the CURRENT EPISODE. When in doubt, do NOT extract

${EXTRACTION_RULES}

Additional rules for messages:
1. Always extract the speaker (the part before the colon ':' in each dialogue line) as the first entity node.
If the speaker is mentioned again in the message, treat both mentions as a **single entity**.
EXCEPTION: if the speaker label is a generic role rather than a name (e.g. "User", "Assistant",
"System", "Bot", "AI", "Human"), do NOT extract it - these are interface labels, not entities.

<EXAMPLES>
<CURRENT EPISODE>
Name: jordan-move
Timestamp: 2025-02-10T18:00:00Z
Source: message
Content: Jordan: We just moved to Denver last month. My spouse started a new role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center.
</CURRENT EPISODE>
Good extractions: "Jordan" (speaker), "Denver" (Location), "Lockheed Martin" (Organization), "Belmont Arts Center" (Location), "ceramics" (Topic)
Do NOT extract: "spouse" (generic reference - extract only if named), "new role" (not an entity), "last month" (temporal), "we" (pronoun)

<CURRENT EPISODE>
Name: nisha-visit
Timestamp: 2025-05-04T12:30:00Z
Source: message
Content: Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park.
</CURRENT EPISODE>
Good extractions: "Nisha" (speaker), "Nisha's dad" (Person), "Riverside Park" (Location)
Do NOT extract: "dad" (bare relational term - qualify as "Nisha's dad"), "dogs" (bare animal word - no specific identity), "next week" (temporal)

<CURRENT EPISODE>
Name: mary-cycling
Timestamp: 2025-06-15T09:00:00Z
Source: message
Content: Mary: I forgot Trigger's leash so I couldn't take him on a dog walk. After that I went road cycling in my new wool coat.
</CURRENT EPISODE>
Good extractions: "Mary" (speaker), "Trigger" (animal name), "dog leash" (Object), "road cycling" (Topic), "wool coat" (Object)
Do NOT extract: "leash" (too generic - use "dog leash"), "cycling" (too generic - use "road cycling"), "coat" (too generic - use "wool coat"), "dog walk" (activity, not an entity)

<CURRENT EPISODE>
Name: nate-gaming
Timestamp: 2025-06-20T22:15:00Z
Source: message
Content: Nate: My gaming room has red and purple lighting and I mostly play on a Gamecube. Last week the windshield on my Mustang got cracked.
</CURRENT EPISODE>
Good extractions: "Nate" (speaker), "gaming room" (Object), "red and purple lighting" (Object), "Gamecube" (Object), "Mustang" (Object), "cracked windshield" (Object)
Do NOT extract: "lighting" (bare head noun - use "red and purple lighting"), "windshield" (bare head noun - use "cracked windshield"), "week" (temporal)

<CURRENT EPISODE>
Name: alex-share
Timestamp: 2025-07-01T20:00:00Z
Source: message
Content: Alex: I shared a pic from the game after the event.
</CURRENT EPISODE>
Good extractions: "Alex" (speaker)
Do NOT extract: "pic" (generic media noun), "game" (generic event noun), "event" (generic event noun)

<CURRENT EPISODE>
Name: jordan-win
Timestamp: 2025-07-05T21:45:00Z
Source: message
Content: Jordan: We won by a tight score. Scoring that last basket felt incredible.
</CURRENT EPISODE>
Good extractions: "Jordan" (speaker)
Do NOT extract: "basket" (ambiguous bare noun that depends on sentence context)
</EXAMPLES>`;

const TEXT_SYSTEM_PROMPT = `You are an entity extraction specialist for unstructured text.
NEVER extract abstract concepts, feelings, or generic words.

Your task is to extract entity nodes that are explicitly mentioned in the CURRENT EPISODE's text.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of
the referenced entities. Only extract distinct entities - if the same real-world entity appears
multiple times, extract it once.

${EXTRACTION_RULES}

Additional rules for documents:
- When in doubt, do NOT extract

<EXAMPLES>
<CURRENT EPISODE>
Name: aan-conference-note
Timestamp: 2025-09-12T15:00:00Z
Source: text
Content: Dr. Amara Osei presented her migraine study results at the AAN conference. The study tracked 340 patients using a new CGRP combination protocol.
</CURRENT EPISODE>
Good extractions: "Dr. Amara Osei" (Person), "AAN" (Organization), "migraine study" (Topic), "CGRP combination protocol" (Object)
Do NOT extract: "results" (generic noun), "340" (number), "patients" (generic noun), "conference" (generic without a specific name)

<CURRENT EPISODE>
Name: alex-event-note
Timestamp: 2025-09-13T10:00:00Z
Source: text
Content: Alex shared a pic after the event and said scoring the last basket felt incredible.
</CURRENT EPISODE>
Good extractions: "Alex" (Person)
Do NOT extract: "pic" (generic media noun), "event" (generic event noun), "basket" (ambiguous bare noun)
</EXAMPLES>`;

const JSON_SYSTEM_PROMPT = `You are an entity extraction specialist for JSON data.
NEVER extract abstract concepts, dates, or generic field values.

Every node you extract becomes a standalone, independently-retrievable memory atom, surfaced to the user
later with none of the surrounding JSON for context. The test behind every rule is: read entirely alone,
later, would this name still identify one specific, distinguishable thing? If not, do NOT extract it.

NEVER extract:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., "true", "active", "pending")
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms (e.g., "spouse", "parent", "pet") - only extract if qualified
  with a possessor name
- Bare generic objects or common nouns (e.g., "supplies", "tickets", "gear") - only extract if
  qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified in the value itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the value itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on surrounding text rather than the extracted value itself

Extract entities from the JSON and classify each using the ENTITY TYPES above.

Guidelines:
1. Extract the primary entity the JSON represents (e.g., a "name" or "user" field).
2. Extract named entities referenced in other properties throughout the JSON structure.
3. Only extract entities specific enough to be uniquely identifiable.
4. Be explicit in naming entities - use full names when available.
5. Use the most specific form present in the data (e.g., "road cycling" not "cycling").
6. If a value would not be meaningful and distinguishable when read alone later, do NOT extract it.

<EXAMPLES>
<CURRENT EPISODE>
Name: jordan-profile
Timestamp: 2024-01-15T09:00:00Z
Source: json
Content: {"user": "Jordan Lee", "company": "Acme Corp", "role": "engineer", "start_date": "2024-01-15", "location": "Denver", "active": true}
</CURRENT EPISODE>
Good extractions: "Jordan Lee" (Person), "Acme Corp" (Organization), "Denver" (Location)
Do NOT extract: "engineer" (role, not an entity), "2024-01-15" (date), "true" (field value)

<CURRENT EPISODE>
Name: alex-post
Timestamp: 2024-02-20T18:30:00Z
Source: json
Content: {"author": "Alex", "attachment_type": "photo", "event_name": "event", "agency": "government"}
</CURRENT EPISODE>
Good extractions: "Alex" (Person)
Do NOT extract: "photo" (generic media noun), "event" (generic event noun), "government" (broad institutional noun)
</EXAMPLES>
`;

const COREFERENCE_INSTRUCTIONS = `<COREFERENCE RESOLUTION>
In unresolvedReferences, report any referring expression in the CURRENT EPISODE - a pronoun (she, they,
it) or a definite description (the manager, the study) - that clearly points to a specific entity but
that you could NOT tie to one of the entities you extracted. Give its surfaceForm and a short
locatingQuote around it. Leave the list empty when every reference resolves.

Example:
Content: The quarterly report was late, so she apologized to the board.
Extracted entities: (none - no person or organization is named)
unresolvedReferences: [{ "surfaceForm": "she", "locatingQuote": "she apologized to the board" }]

For each extracted entity, also provide:
- identifyingDescription: a short standalone descriptor that pins down the entity even where its name does not
appear - its role or type plus the most distinctive facts stated here. Write it so a later reader
holding only this descriptor could tell which entity a pronoun refers to. State gender only when the
text itself establishes it (a pronoun, a stated fact) - NEVER inferred from the name.
- aliases: EVERY distinctive surface form this text uses for the entity, other than pronouns.
Include both name variants (e.g. "Dr. Osei", "Osei") AND the role or definite descriptions the
text uses for it (e.g. "the captain", "the first officer", "the engineer", "the trawler"). This
matters: later text may refer to the entity only by such a description, and the alias is how it
is recognized as the same entity. Only include a description when it clearly points to this one
entity here - skip it if it could refer to more than one.
- referredToAsPronouns: the pronouns THIS text itself uses for the entity, in subject form
(he, she, it, they). Observed usage only - NEVER infer a pronoun from the entity's name or type.
Include this field only when the text uses a pronoun for the entity; omit it otherwise.

Example (the coref fields for two entities):
Content: The captain, Captain Ruiz, had no faith in the sonar. The first officer was a quiet
Peruvian named Serrano, and he drew up the watch bill.
- "Captain Ruiz": identifyingDescription "ship captain who distrusts the sonar", aliases ["the captain"]
- "Serrano": identifyingDescription "first officer, a quiet Peruvian", aliases ["the first officer"], referredToAsPronouns ["he"]
</COREFERENCE RESOLUTION>`;

function buildSystemPrompt(source: EpisodeType): string {
  let base: string;
  switch (source) {
    case EpisodeType.message:
      base = MESSAGE_SYSTEM_PROMPT;
      break;
    case EpisodeType.json:
      base = JSON_SYSTEM_PROMPT;
      break;
    default:
      base = TEXT_SYSTEM_PROMPT;
  }
  return `${base}\n\n${COREFERENCE_INSTRUCTIONS}`;
}

export function buildExtractNodesMessages(ctx: {
  episode: EpisodicNode;
  previousEpisodes: EpisodicNode[];
  entityTypes?: EntityTypeMap;
  customInstructions?: string;
}): BaseMessage[] {
  const { episode, previousEpisodes, entityTypes, customInstructions } = ctx;

  const systemPrompt = buildSystemPrompt(episode.source);

  const previousEpisodesText = formatPreviousEpisodes(previousEpisodes);

  let humanContent = `Apply every rule from the system instructions when extracting entities from the episode below.

<PREVIOUS EPISODES>
${previousEpisodesText}
</PREVIOUS EPISODES>

<CURRENT EPISODE>
${formatCurrentEpisode(episode, { includeSource: true })}
</CURRENT EPISODE>`;

  humanContent += `\n\n<SOURCE DESCRIPTION>\n${episode.sourceDescription}\n</SOURCE DESCRIPTION>`;

  if (entityTypes && Object.keys(entityTypes).length > 0) {
    const entityTypesText = Object.entries(entityTypes)
      .map(
        ([label, { description }], index) =>
          `{id: ${index}, label: "${label}", description: "${description}"}`,
      )
      .join('\n');
    humanContent += `\n\n<ENTITY TYPES>\n${entityTypesText}\n</ENTITY TYPES>`;
  }

  if (customInstructions) {
    humanContent += `\n\n<CUSTOM INSTRUCTIONS>\n${customInstructions}\n</CUSTOM INSTRUCTIONS>`;
  }
  return [new SystemMessage(systemPrompt), new HumanMessage(humanContent)];
}

// Bare pronouns don't identify an entity, so aliases must exclude them (a
// bare-pronoun alias seeds the recency mis-binding in known-failures Failure 1).
// Scoped to pronoun classes that corefer to a definite entity, plus the
// zero-collision reflexives/reciprocals. Indefinite pronouns (someone, nothing,
// one, ...) are deliberately omitted: non-referential, and several double as
// proper names ("One", "Nothing", "Others").
const BARE_PRONOUNS = new Set([
  // personal (subjective + objective)
  'i',
  'you',
  'he',
  'she',
  'it',
  'we',
  'they',
  'me',
  'him',
  'her',
  'us',
  'them',
  // possessive (pronouns + determiners)
  'my',
  'mine',
  'your',
  'yours',
  'his',
  'hers',
  'its',
  'our',
  'ours',
  'their',
  'theirs',
  // demonstrative
  'this',
  'that',
  'these',
  'those',
  // interrogative + relative
  'who',
  'whom',
  'whose',
  'which',
  'what',
  'whoever',
  'whomever',
  'whichever',
  'whatever',
  // reflexive + intensive
  'myself',
  'yourself',
  'yourselves',
  'himself',
  'herself',
  'itself',
  'ourselves',
  'themselves',
  // reciprocal
  'each other',
  'one another',
]);

export function buildExtractNodesValidator(ctx: {
  entityTypes?: EntityTypeMap;
}): (parsed: ExtractedEntitiesOutput) => Violation[] {
  const typeCount = ctx.entityTypes ? Object.keys(ctx.entityTypes).length : 0;

  return (parsed) => {
    const violations: Violation[] = [];

    parsed.extractedEntities.forEach((e, i) => {
      for (const alias of e.aliases) {
        if (BARE_PRONOUNS.has(alias.trim().toLowerCase())) {
          violations.push({
            code: 'extract-nodes.bare-pronoun-alias',
            message: `extractedEntities[${i}] has a bare pronoun in aliases; aliases must exclude bare pronouns`,
          });
        }
      }
    });

    if (typeCount > 0) {
      for (const e of parsed.extractedEntities) {
        if (e.entityTypeId === undefined) continue;
        if (e.entityTypeId < 0 || e.entityTypeId >= typeCount) {
          violations.push({
            code: 'extract-nodes.entity-type-id-out-of-range',
            message: `entityTypeId ${e.entityTypeId} for "${e.name}" is out of range [0, ${typeCount})`,
          });
        }
      }
    }
    return violations;
  };
}
