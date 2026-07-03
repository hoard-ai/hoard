import { isDefaultExportSpan, LangfuseSpanProcessor } from '@langfuse/otel';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { NodeSDK } from '@opentelemetry/sdk-node';
import {
  ConsoleSpanExporter,
  type ReadableSpan,
  type Span as SdkSpan,
  SimpleSpanProcessor,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { PrismaInstrumentation } from '@prisma/instrumentation';

import { parseLangfuseConfig } from '@/config/langfuse';
import { parseOtelConfig } from '@/config/otel';

import { setLangfuseEnabled } from './observability-utils';

const HOARD_SCOPE = 'hoard';
const PRISMA_SCOPE = 'prisma';
const OBSERVATION_LEVEL_ATTR = 'langfuse.observation.level';

/**
 * Ref-counts in-flight `hoard` spans per trace id so `shouldExportSpan`
 * can gate Prisma spans on hoard-rooted traces, and tags Prisma spans
 * with `level=DEBUG` so Langfuse hides them behind the DEBUG filter.
 */
class HoardTraceTracker implements SpanProcessor {
  private readonly counts = new Map<string, number>();

  has(traceId: string): boolean {
    return this.counts.has(traceId);
  }

  onStart(span: SdkSpan): void {
    const scope = span.instrumentationScope.name;
    if (scope === PRISMA_SCOPE) {
      span.setAttribute(OBSERVATION_LEVEL_ATTR, 'DEBUG');
      return;
    }
    if (scope !== HOARD_SCOPE) return;
    const traceId = span.spanContext().traceId;
    this.counts.set(traceId, (this.counts.get(traceId) ?? 0) + 1);
  }

  onEnd(span: ReadableSpan): void {
    if (span.instrumentationScope.name !== HOARD_SCOPE) return;
    const traceId = span.spanContext().traceId;
    const next = (this.counts.get(traceId) ?? 0) - 1;
    if (next <= 0) this.counts.delete(traceId);
    else this.counts.set(traceId, next);
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}

/**
 * Bootstraps the OpenTelemetry NodeSDK with any configured span processors.
 *
 * MUST run BEFORE NestJS / LangChain modules are imported so auto-instrumentation
 * hooks can attach. NestJS hasn't booted yet, so we cannot use `ConfigService` -
 * instead we call the canonical Zod parsers directly.
 *
 * ---------------------------------------------------------------------------
 * TODO(prod): OpenTelemetry Collector (otel-collector.yaml)
 *
 * Filtering / sampling / PII scrubbing belong at the Collector, not in-process.
 * Tail-based sampling lets you keep the FULL trace whenever any span has ERROR
 * status - successful siblings included.
 *
 *   app --OTLP--> [Collector sidecar] --> CloudWatch / X-Ray / Sentry / S3
 *
 * When that day comes, swap `ConsoleSpanExporter` here for `OTLPTraceExporter`
 * ---------------------------------------------------------------------------
 */
export function startOtel(): { shutdown: () => Promise<void> } {
  const langfuseConfig = parseLangfuseConfig();
  const otelConfig = parseOtelConfig();

  if (langfuseConfig.enabled && !otelConfig.telemetryEnabled) {
    throw new Error(
      'LANGFUSE_ENABLED=true requires TELEMETRY_ENABLED=true. ' +
        'Langfuse exports observations through the OTel SDK; the SDK must be running.',
    );
  }

  // When telemetry is off we skip SDK registration entirely. `@Span`-decorated
  // methods still call `trace.getTracer(...)`, but with no SDK registered OTel
  // returns its built-in no-op tracer, so spans are created-but-discarded and
  // nothing is exported.
  if (!otelConfig.telemetryEnabled) {
    return { shutdown: () => Promise.resolve() };
  }
  const processors: SpanProcessor[] = [];

  if (langfuseConfig.enabled) {
    setLangfuseEnabled(true);
    const traceTracker = new HoardTraceTracker();
    processors.push(traceTracker);

    processors.push(
      new LangfuseSpanProcessor({
        publicKey: langfuseConfig.publicKey,
        secretKey: langfuseConfig.secretKey,
        baseUrl: langfuseConfig.baseUrl,
        environment: langfuseConfig.environment,
        // `batched` is required for long-running servers like this one
        exportMode: 'batched',
        // Default filter keeps only LLM-focused spans (langfuse-sdk scope,
        // gen_ai.* attrs, known LLM instrumentation). Also let through our
        // own `hoard` scope so manually-traced pipeline spans appear
        // as parents of the LangChain generations, plus Prisma spans whose
        // trace originated from a `hoard` span
        shouldExportSpan: ({ otelSpan }) => {
          const scope = otelSpan.instrumentationScope.name;
          if (isDefaultExportSpan(otelSpan)) return true;
          if (scope === HOARD_SCOPE) return true;
          if (scope === PRISMA_SCOPE) {
            return traceTracker.has(otelSpan.spanContext().traceId);
          }
          return false;
        },
      }),
    );
  }
  if (otelConfig.consoleExportEnabled) {
    processors.push(new SimpleSpanProcessor(new ConsoleSpanExporter()));
  }
  if (processors.length === 0) {
    return { shutdown: () => Promise.resolve() };
  }

  const sdk = new NodeSDK({
    serviceName: 'hoard-api',
    spanProcessors: processors,
    instrumentations: [
      getNodeAutoInstrumentations({
        '@opentelemetry/instrumentation-fs': { enabled: false },

        // Disabled because Express 5 (and Nest 11's Express adapter)
        // uses the standalone `router` package internally - leaving it
        // enabled wraps every Express middleware a second time as `middleware -
        // patched`, with no extra info beyond the named express spans.
        '@opentelemetry/instrumentation-router': { enabled: false },

        // Disabled to keep prompts/outputs out of OTel spans entirely. When we
        // add OpenAI as a vendor, all LLM observability flows through the
        // `LlmTracer` abstraction (Langfuse when enabled, no-op otherwise) so the
        // LANGFUSE_ENABLED toggle stays the single source of truth.
        '@opentelemetry/instrumentation-openai': { enabled: false },

        // Ignore health check endpoints
        '@opentelemetry/instrumentation-http': {
          enabled: true,
          ignoreIncomingRequestHook: (req) => {
            return req.url?.includes('/health') || req.url?.includes('/metrics') || false;
          },
        },

        // Default `db.statement` is `<cmd> <args...>`, which leaks session IDs
        // on `GET sess:<id>` and the full serialized session body on `SET`
        // The session ID itself is a bearer credential, so we redact it;
        // for all other commands we keep `<cmd> <key>` but never values.
        '@opentelemetry/instrumentation-redis': {
          enabled: true,
          dbStatementSerializer: (cmdName, cmdArgs) => {
            const firstArg = cmdArgs[0];
            if (typeof firstArg !== 'string') return cmdName;
            if (firstArg.startsWith('sess:')) return `${cmdName} sess:<redacted>`;
            return `${cmdName} ${firstArg}`;
          },
        },
      }),
      new PrismaInstrumentation(),
    ],
  });
  sdk.start();

  const shutdown = (): Promise<void> =>
    sdk
      .shutdown()
      .then(() => console.log('Tracing terminated'))
      .catch((error) => console.log('Error terminating tracing', error))
      .finally(() => process.exit(0));

  process.once('SIGTERM', () => void shutdown());
  process.once('SIGINT', () => void shutdown());

  return { shutdown };
}
