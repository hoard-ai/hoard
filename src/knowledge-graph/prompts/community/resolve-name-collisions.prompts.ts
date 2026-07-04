import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { NodeNameSchema } from '@/knowledge-graph/types';
import type { Violation } from '@/llm';

// Schema

const ResolvedNameSchema = z.object({
  tempId: z.int().nonnegative().describe('Echo of the input tempId for the collider'),
  name: NodeNameSchema.describe(
    'New, distinct noun-phrase label (2-6 words) for this collider',
  ),
});

export const ResolveNameCollisionsSchema = z.object({
  resolutions: z
    .array(ResolvedNameSchema)
    .describe('One resolution per input collider, in any order'),
});

export type ResolveNameCollisionsOutput = z.infer<typeof ResolveNameCollisionsSchema>;

// Prompt builder

export type Collider = { tempId: number; summary: string };

const SYSTEM_PROMPT = `You label clusters of related entities produced by automatic community detection on a knowledge graph, 
resolving collisions where multiple clusters received the same first-pass label.

You will receive a set of COLLIDERS - clusters whose labels are identical or near-identical - and a list of NAMES IN USE that 
are already taken by other communities in the same graph. Each cluster is the output of an automatic Louvain partition and is 
NOT a human-curated topic.

Your job is to emit a new label for each collider that fits the same style as the first-pass namer: 
a 2-to-6-word noun phrase with proper nouns in Title Case and common words lowercase.

Rules:
- Output one resolution per input collider. Each output 'tempId' MUST echo an input tempId exactly. NEVER drop or add ids.
- Each new name is a single noun phrase, 2 to 6 words. NEVER write a sentence.
- Each new name MUST be distinct from every entry in NAMES IN USE AND distinct from every other name you emit in this call. 
Compare case-insensitively and after trimming.
- Prefer differentiating by content. Read each collider's summary and lead with the most distinguishing concrete token 
(a person, place, project, named topic, time qualifier) that the summary actually supports. 
This is the primary strategy whenever the summaries differ in any meaningful way.
- Avoid bare generic headers like "Health", "Family", "Work", "Hobbies" - they collide easily and convey nothing.
- Numbering convention for near-identical content ONLY: if two or more colliders cover effectively the 
same content and no distinguishing token exists in any summary, emit names that share a common root and 
append the fixed parenthetical suffix " (2)", " (3)", " (4)", ... in input order. The first collider keeps the 
bare root with no suffix; numbering starts at "(2)". NEVER use "(1)".
- Numbering applies WITHIN a single collision group. Do NOT number across unrelated colliders.
- NEVER include quotes, trailing punctuation, leading articles ("The ", "A ", "An "), or meta-language like "Community of", "Group of", "Cluster of".
- NEVER mention the summarization process, the graph, nodes, edges, clusters, or community detection in the label itself.
- Use Title Case for proper nouns; lowercase common words.

<EXAMPLES>
<COLLIDERS>
- tempId: 0, summary: "Jordan Lee teaches beginner ceramics at Belmont Arts Center on Wednesday evenings; the kiln room opened in March 2025."
- tempId: 1, summary: "Mina runs an advanced wheel-thrown workshop series in Brooklyn out of Greenpoint Clay Studio; 18 attendees in April 2025."
</COLLIDERS>
<NAMES IN USE>
- Denver tech meetup
</NAMES IN USE>
Result: {"resolutions": [{"tempId": 0, "name": "Belmont Arts Center ceramics"}, {"tempId": 1, "name": "Greenpoint Clay Studio workshops"}]}
(differentiating by the named studio in each summary; no numbering needed)

<COLLIDERS>
- tempId: 0, summary: "User cooks dinner at home most weeknights; tracks recipes in a shared notebook."
- tempId: 1, summary: "User cooks dinner at home most weeknights; tracks recipes in a shared notebook."
</COLLIDERS>
<NAMES IN USE>
None
</NAMES IN USE>
Result: {"resolutions": [{"tempId": 0, "name": "weeknight home cooking"}, {"tempId": 1, "name": "weeknight home cooking (2)"}]}
(summaries are effectively identical - fall back to the numbering convention, starting at (2))`;

function formatColliders(colliders: readonly Collider[]): string {
  return colliders
    .map((c) => `- tempId: ${c.tempId}, summary: "${c.summary}"`)
    .join('\n');
}

export type ResolveNameCollisionsCtx = {
  colliders: readonly Collider[];
  namesInUse: readonly string[];
};

export function buildResolveNameCollisionsMessages(
  ctx: ResolveNameCollisionsCtx,
): BaseMessage[] {
  const { colliders, namesInUse } = ctx;

  const namesInUseText =
    namesInUse.length === 0 ? 'None' : namesInUse.map((n) => `- ${n}`).join('\n');

  const humanContent = `Apply every rule from the system instructions when renaming the colliders below.

<COLLIDERS>
${formatColliders(colliders)}
</COLLIDERS>

<NAMES IN USE>
${namesInUseText}
</NAMES IN USE>

Your response MUST include EXACTLY ${colliders.length} ${colliders.length === 1 ? 'resolution' : 'resolutions'}, one per input collider, echoing each input tempId exactly.`;

  return [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(humanContent)];
}

const norm = (s: string): string => s.trim().toLowerCase();

export function buildResolveNameCollisionsValidator(
  ctx: ResolveNameCollisionsCtx,
): (parsed: ResolveNameCollisionsOutput) => Violation[] {
  const validTempIds = new Set(ctx.colliders.map((c) => c.tempId));
  const expectedCount = ctx.colliders.length;
  const namesInUseNorm = new Set(ctx.namesInUse.map(norm));

  return (parsed) => {
    const violations: Violation[] = [];
    const { resolutions } = parsed;

    if (resolutions.length !== expectedCount) {
      violations.push({
        code: 'resolve-name-collisions.wrong-resolution-count',
        message: `expected ${expectedCount} resolutions, got ${resolutions.length}`,
      });
    }

    const seenTempIds = new Set<number>();
    const seenNames = new Set<string>();
    for (const r of resolutions) {
      if (!validTempIds.has(r.tempId)) {
        violations.push({
          code: 'resolve-name-collisions.unknown-temp-id',
          message: `tempId ${r.tempId} was not in the input collider set`,
        });
      }
      if (seenTempIds.has(r.tempId)) {
        violations.push({
          code: 'resolve-name-collisions.duplicate-temp-id',
          message: `duplicate tempId ${r.tempId} in resolutions`,
        });
      }
      seenTempIds.add(r.tempId);

      const n = norm(r.name);
      if (namesInUseNorm.has(n)) {
        violations.push({
          code: 'resolve-name-collisions.collides-with-existing',
          message: `name "${r.name}" collides with an existing NAMES IN USE entry`,
        });
      }
      if (seenNames.has(n)) {
        violations.push({
          code: 'resolve-name-collisions.duplicate-name',
          message: `duplicate name "${r.name}" across resolutions`,
        });
      }
      seenNames.add(n);
    }
    return violations;
  };
}
