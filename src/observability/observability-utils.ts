import { type Attributes, trace } from '@opentelemetry/api';

export const TRACER_NAME = 'hoard';

// Langfuse OTel attribute keys, mapped at ingestion per
// https://langfuse.com/integrations/native/opentelemetry (attribute mapping).
export const OBSERVATION_TYPE_ATTR = 'langfuse.observation.type';
export const OBSERVATION_LEVEL_ATTR = 'langfuse.observation.level';
export const OBSERVATION_INPUT_ATTR = 'langfuse.observation.input';
export const OBSERVATION_OUTPUT_ATTR = 'langfuse.observation.output';
export const TRACE_NAME_ATTR = 'langfuse.trace.name';
export const TRACE_INPUT_ATTR = 'langfuse.trace.input';
export const TRACE_OUTPUT_ATTR = 'langfuse.trace.output';

/**
 * Tracks whether `otel.ts` registered the Langfuse span processor at boot.
 *
 * The `@Span` decorator reads this to decide whether to serialize I/O
 *
 * `otel.ts` is the single source of truth for whether Langfuse is wired up,
 * so it's also the single place that flips this flag.
 */
let langfuseEnabled = false;

export function setLangfuseEnabled(value: boolean): void {
  langfuseEnabled = value;
}

export function isLangfuseEnabled(): boolean {
  return langfuseEnabled;
}

/**
 * Records a discrete occurrence as a zero-duration child span of the active
 * span, typed as a Langfuse `event` observation. One record per call - unlike
 * a span attribute, concurrent occurrences under the same parent don't
 * overwrite each other.
 */
export function recordSpanEvent(name: string, attributes: Attributes): void {
  trace
    .getTracer(TRACER_NAME)
    .startSpan(name, {
      attributes: {
        ...attributes,
        [OBSERVATION_TYPE_ATTR]: 'event',
        [OBSERVATION_LEVEL_ATTR]: 'DEBUG',
      },
    })
    .end();
}
