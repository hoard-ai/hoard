/**
 * Ported from `nestjs-otel` on 2026-05-16
 * (https://github.com/pragmaticivan/nestjs-otel - Apache 2.0). The
 * `asLangfuseTrace` option is a local extension, and the generic types were
 * tightened (`unknown[]` instead of `any[]`, precise descriptor / constructor
 * types) to satisfy `@typescript-eslint`'s no-unsafe-* rules without
 * suppressions. Everything else is verbatim apart from adapting imports.
 */
import { type Span as ApiSpan, SpanStatusCode, trace } from '@opentelemetry/api';
import 'reflect-metadata';

import { isLangfuseEnabled } from '../observability-utils';
import type { ExtendedSpanOptions, ObservationKind } from '../types';
import { safeStringify } from './serialize';

const TRACER_NAME = 'hoard';

const OBSERVATION_TYPE_ATTR = 'langfuse.observation.type';
const OBSERVATION_INPUT_ATTR = 'langfuse.observation.input';
const OBSERVATION_OUTPUT_ATTR = 'langfuse.observation.output';
const TRACE_INPUT_ATTR = 'langfuse.trace.input';
const TRACE_OUTPUT_ATTR = 'langfuse.trace.output';

const kindAttribute = (kind: ObservationKind | undefined): string | undefined =>
  kind && kind !== 'span' ? kind : undefined;

function copyMetadata(from: object, to: object): void {
  for (const key of Reflect.getMetadataKeys(from)) {
    Reflect.defineMetadata(key, Reflect.getMetadata(key, from), to);
  }
}

type AnyMethod = (this: unknown, ...args: unknown[]) => unknown;

const recordException = (span: ApiSpan, error: unknown): void => {
  const err = error instanceof Error ? error : new Error(String(error));
  span.recordException(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
};

const handleOnResult = (
  span: ApiSpan,
  onResult: ExtendedSpanOptions['onResult'],
  result: unknown,
): void => {
  if (!onResult) return;
  try {
    const attrs = onResult(result);
    if (attrs?.attributes) {
      span.setAttributes(attrs.attributes);
    }
  } catch (error) {
    recordException(span, error);
  }
};

type SpanDecoratorOptions<TArgs extends unknown[]> =
  | ExtendedSpanOptions
  | ((...args: TArgs) => ExtendedSpanOptions);

type SpanMethodDecorator<TArgs extends unknown[], TReturn> = (
  target: object,
  propertyKey: PropertyKey,
  descriptor: TypedPropertyDescriptor<(...args: TArgs) => TReturn>,
) => void;

export function Span<TArgs extends unknown[], TReturn>(
  options?: SpanDecoratorOptions<TArgs>,
): SpanMethodDecorator<TArgs, TReturn>;
export function Span<TArgs extends unknown[], TReturn>(
  name?: string,
  options?: SpanDecoratorOptions<TArgs>,
): SpanMethodDecorator<TArgs, TReturn>;
export function Span<TArgs extends unknown[], TReturn>(
  nameOrOptions?: string | SpanDecoratorOptions<TArgs>,
  maybeOptions?: SpanDecoratorOptions<TArgs>,
): SpanMethodDecorator<TArgs, TReturn> {
  return (target, propertyKey, descriptor) => {
    const explicitName = typeof nameOrOptions === 'string';
    const name = explicitName
      ? nameOrOptions
      : `${target.constructor.name}.${String(propertyKey)}`;
    const options: SpanDecoratorOptions<TArgs> =
      (explicitName ? maybeOptions : nameOrOptions) ?? {};

    const originalFunction = descriptor.value;
    if (typeof originalFunction !== 'function') {
      throw new Error(
        `The @Span decorator can be only used on functions, but ${propertyKey.toString()} is not a function.`,
      );
    }

    const wrappedFunction: AnyMethod = function (this, ...args) {
      const tracer = trace.getTracer(TRACER_NAME);

      const spanOptions =
        typeof options === 'function' ? options(...(args as TArgs)) : options;

      const { onResult, asLangfuseTrace, observationKind, captureIo, ...otelOptions } =
        spanOptions;
      const shouldCaptureIo = captureIo !== false && isLangfuseEnabled();

      return tracer.startActiveSpan(name, otelOptions, (span): unknown => {
        if (asLangfuseTrace) {
          span.setAttribute('langfuse.trace.name', name);
        }
        const kindAttr = kindAttribute(observationKind);
        if (kindAttr) {
          span.setAttribute(OBSERVATION_TYPE_ATTR, kindAttr);
        }
        // `isRecording()` is false when no SDK is registered (telemetry off),
        // so serialization is skipped entirely on the no-op path.
        if (shouldCaptureIo && span.isRecording()) {
          const inputStr = safeStringify(args);
          span.setAttribute(OBSERVATION_INPUT_ATTR, inputStr);
          // Langfuse derives trace input/output from the root observation, but
          // only when the SDK can identify that observation as the root. Setting
          // `langfuse.trace.input/output` directly removes that dependency and
          // guarantees the Langfuse trace gets populated.
          if (asLangfuseTrace) {
            span.setAttribute(TRACE_INPUT_ATTR, inputStr);
          }
        }
        try {
          const result = (originalFunction as AnyMethod).apply(this, args);

          if (result instanceof Promise) {
            return result
              .then((res: unknown) => {
                handleOnResult(span, onResult, res);
                if (shouldCaptureIo && span.isRecording()) {
                  const outputStr = safeStringify(res);
                  span.setAttribute(OBSERVATION_OUTPUT_ATTR, outputStr);
                  if (asLangfuseTrace) {
                    span.setAttribute(TRACE_OUTPUT_ATTR, outputStr);
                  }
                }
                return res;
              })
              .catch((error: unknown) => {
                recordException(span, error);
                throw error;
              })
              .finally(() => {
                span.end();
              });
          }

          handleOnResult(span, onResult, result);
          if (shouldCaptureIo && span.isRecording()) {
            const outputStr = safeStringify(result);
            span.setAttribute(OBSERVATION_OUTPUT_ATTR, outputStr);
            if (asLangfuseTrace) {
              span.setAttribute(TRACE_OUTPUT_ATTR, outputStr);
            }
          }
          span.end();
          return result;
        } catch (error) {
          recordException(span, error);
          span.end();
          throw error;
        }
      });
    };

    // Proxy preserves the original function's `name`/`length` so frameworks
    // that introspect them (NestJS routing, OpenAPI generators) keep working.
    const proxy = new Proxy(originalFunction, {
      apply: (_target, thisArg, argArray: unknown[]) =>
        wrappedFunction.apply(thisArg, argArray),
    });
    descriptor.value = proxy as (...args: TArgs) => TReturn;

    copyMetadata(originalFunction, descriptor.value);
  };
}
