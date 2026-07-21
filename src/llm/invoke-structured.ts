import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { DEFAULT_MAX_RETRIES, type InvokeStructuredOptions } from './types';

// TODO: We do not handle this error in any way.
// `issues` carries only structural identifiers: `path: zod-code` for schema
// failures, and validator-defined codes for invariant failures. User-derived
// text from validator/Zod messages is fed back to the model in the retry
// HumanMessage but never stored or surfaced here.
export class StructuredOutputValidationError extends Error {
  constructor(
    public readonly runName: string,
    public readonly issues: readonly string[],
  ) {
    super(
      `Structured output validation failed for "${runName}" after retries: ${issues.slice(0, 3).join('; ')}`,
    );
    this.name = 'StructuredOutputValidationError';
  }
}

export async function invokeStructured<S extends z.ZodType>(
  model: BaseChatModel,
  schema: S,
  messages: BaseMessage[],
  opts: InvokeStructuredOptions<z.infer<S>>,
): Promise<z.infer<S>> {
  const {
    runName,
    tags = [],
    callbacks,
    maxRetries = DEFAULT_MAX_RETRIES,
    validate,
  } = opts;
  const jsonSchema = z.toJSONSchema(schema, { io: 'input' });
  const runnable = model.withStructuredOutput(jsonSchema);

  let currentMessages = messages;
  let lastCodes: string[] = [];

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const raw: unknown = await runnable.invoke(currentMessages, {
      callbacks,
      runName: attempt === 1 ? runName : `${runName}.retry-${attempt - 1}`,
      tags: attempt === 1 ? tags : [...tags, 'retry'],
    });

    const parsed = await schema.safeParseAsync(raw);
    if (!parsed.success) {
      lastCodes = zodCodes(parsed.error);
      // Echo the model's own rejected output back as an assistant turn so the
      // retry can refine it rather than regenerate blind from the feedback alone.
      currentMessages = [
        ...currentMessages,
        new AIMessage(JSON.stringify(raw, null, 2)),
        new HumanMessage(formatSchemaFeedback(parsed.error)),
      ];
      continue;
    }
    const violations = validate?.(parsed.data) ?? [];
    if (violations.length === 0) return parsed.data;

    lastCodes = violations.map((v) => v.code);
    currentMessages = [
      ...currentMessages,
      new AIMessage(JSON.stringify(raw, null, 2)),
      new HumanMessage(formatInvariantFeedback(violations.map((v) => v.message))),
    ];
  }
  throw new StructuredOutputValidationError(runName, lastCodes);
}

function zodCodes(error: z.ZodError): string[] {
  return error.issues.map(
    (i) => `${i.path.length ? i.path.join('.') : '(root)'}: ${i.code}`,
  );
}

function formatSchemaFeedback(error: z.ZodError): string {
  let out =
    'Your previous response did not match the required schema. Validation errors:\n';
  out += z.prettifyError(error);
  out += '\n\nPlease retry with a response that resolves them.';
  return out;
}

function formatInvariantFeedback(messages: readonly string[]): string {
  const bullets = messages.map((v) => `- ${v}`).join('\n');
  let out = `Your previous response parsed but violated these output invariants:\n${bullets}`;
  out += '\n\nPlease retry with a response that resolves them.';
  return out;
}
