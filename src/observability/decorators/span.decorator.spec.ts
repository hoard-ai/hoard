/**
 * Ported from `nestjs-otel` on 2026-05-16
 * (https://github.com/pragmaticivan/nestjs-otel - Apache 2.0). The
 * `asLangfuseTrace` cases are local additions, and the test fixtures were
 * re-typed (precise `FailingThenable` type, no `any`, no `expect.anything()`)
 * to satisfy `@typescript-eslint`'s no-unsafe-* rules without suppressions.
 * Everything else is verbatim apart from style adjustments.
 */
import { SetMetadata } from '@nestjs/common';
import { SpanKind, SpanStatusCode } from '@opentelemetry/api';
import {
  InMemorySpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-node';
import 'reflect-metadata';

import { isLangfuseEnabled, setLangfuseEnabled } from '../observability-utils';
import { Span } from './span.decorator';

const TestDecoratorThatSetsMetadata = (): MethodDecorator =>
  SetMetadata('some-metadata', true);

const symbol = Symbol('testSymbol');

class TestSpan {
  @Span()
  singleSpan() {}

  @Span()
  doubleSpan() {
    return this.singleSpan();
  }

  @Span('foo', { kind: SpanKind.PRODUCER })
  fooProducerSpan() {}

  @Span('bar', (a, b) => ({ attributes: { a, b } }))
  argsInOptions(_a: number, _b: string) {}

  @Span({ kind: SpanKind.PRODUCER })
  implicitSpanNameWithOptions() {}

  @Span((a, b) => ({ attributes: { a, b } }))
  argsInOptionsWithImplicitName(_a: number, _b: string) {}

  @Span()
  error() {
    throw new Error('hello world');
  }

  @Span()
  @TestDecoratorThatSetsMetadata()
  metadata() {}

  // `@Span()` is applied to `[symbol]` after class definition (see below) — the
  // trivago import-sort plugin's babel parser is configured with
  // `decorators-legacy`, which doesn't accept decorators on computed method names.
  [symbol]() {}

  @Span({
    onResult: (result) => ({ attributes: { result: result as string } }),
  })
  syncMethod() {
    return 'success';
  }

  @Span({
    onResult: (result) => ({ attributes: { result: result as string } }),
  })
  asyncMethod(): Promise<string> {
    return Promise.resolve('async success');
  }

  @Span({
    onResult: (_result) => {
      throw new Error('onResult error');
    },
  })
  errorInOnResult() {
    return 'success';
  }

  @Span()
  asyncError(): Promise<never> {
    return Promise.reject(new Error('async hello world'));
  }

  lastLazyThenable?: LazyThenable<string>;

  @Span()
  returnsLazyThenable() {
    this.lastLazyThenable = makeLazyThenable('lazy result');
    return this.lastLazyThenable;
  }

  @Span()
  returnsFailingLazyThenable() {
    return makeFailingLazyThenable(new Error('lazy rejection'));
  }

  @Span({
    onResult: (result) => ({
      attributes: { count: (result as { rows: number[] }).rows.length },
    }),
  })
  lazyThenableOnResult() {
    return makeLazyThenable({ rows: [1, 2, 3] });
  }

  @Span({ asLangfuseTrace: true })
  langfuseTraceImplicitName() {}

  @Span('explicitTraceName', { asLangfuseTrace: true })
  langfuseTraceExplicitName() {}

  @Span()
  notLangfuseTrace() {}

  @Span('retriever.search', { observationKind: 'retriever' })
  retrieverKind() {}

  @Span('embedding.embed', { observationKind: 'embedding' })
  embeddingKind() {}

  @Span('plain.span', { observationKind: 'span' })
  spanKind() {}

  @Span('captures.io')
  capturesIo(_arg: { query: string }): { rows: number[] } {
    return { rows: [1, 2, 3] };
  }

  @Span('captures.io.opt-out', { captureIo: false })
  capturesIoOptOut(_arg: { query: string }) {
    return 'should-not-be-captured';
  }

  @Span('captures.io.throws')
  capturesIoThrows(_arg: { query: string }): never {
    throw new Error('boom');
  }

  @Span('captures.io.async')
  capturesIoAsync(arg: { query: string }): Promise<{ rows: number[] }> {
    return Promise.resolve({ rows: [arg.query.length] });
  }

  @Span('langfuse.trace.io', { asLangfuseTrace: true })
  langfuseTraceCapturesIo(_arg: { query: string }): { rows: number[] } {
    return { rows: [1, 2, 3] };
  }

  @Span('langfuse.trace.io.opt-out', { asLangfuseTrace: true, captureIo: false })
  langfuseTraceOptedOutOfIo(_arg: { query: string }): { rows: number[] } {
    return { rows: [1, 2, 3] };
  }
}

const symbolDescriptor = Object.getOwnPropertyDescriptor(
  TestSpan.prototype,
  symbol,
) as TypedPropertyDescriptor<() => void>;
Span()(TestSpan.prototype, symbol, symbolDescriptor);
Object.defineProperty(TestSpan.prototype, symbol, symbolDescriptor);

type LazyThenable<T> = PromiseLike<T> & {
  triggered: boolean;
  catch: (onRejected?: (e: unknown) => unknown) => PromiseLike<unknown>;
};

function makeLazyThenable<T>(value: T): LazyThenable<T> {
  const thenable: LazyThenable<T> = {
    triggered: false,
    then(onFulfilled, onRejected) {
      thenable.triggered = true;
      try {
        const result = onFulfilled ? onFulfilled(value) : (value as never);
        return Promise.resolve(result);
      } catch (error) {
        const reason = error instanceof Error ? error : new Error(String(error));
        if (onRejected) {
          return Promise.resolve(onRejected(reason)) as never;
        }
        return Promise.reject(reason) as never;
      }
    },
    catch(onRejected) {
      return thenable.then(undefined, onRejected);
    },
  };
  return thenable;
}

type FailingThenable = PromiseLike<never> & {
  triggered: boolean;
  catch: (onRejected?: (e: unknown) => unknown) => PromiseLike<never>;
};

function makeFailingLazyThenable(error: Error): FailingThenable {
  const thenable: FailingThenable = {
    triggered: false,
    then(_onFulfilled, onRejected) {
      thenable.triggered = true;
      if (onRejected) {
        return Promise.resolve(onRejected(error)) as never;
      }
      return Promise.reject(error) as never;
    },
    catch(onRejected) {
      return thenable.then(undefined, onRejected) as PromiseLike<never>;
    },
  };
  return thenable;
}

describe('Span', () => {
  let instance: TestSpan;
  let traceExporter: InMemorySpanExporter;
  let spanProcessor: SimpleSpanProcessor;
  let provider: NodeTracerProvider;

  beforeAll(() => {
    instance = new TestSpan();
    traceExporter = new InMemorySpanExporter();
    spanProcessor = new SimpleSpanProcessor(traceExporter);
    provider = new NodeTracerProvider({
      spanProcessors: [spanProcessor],
    });
    provider.register();
  });

  afterEach(async () => {
    await spanProcessor.forceFlush();
    traceExporter.reset();
  });

  afterAll(async () => {
    await provider.shutdown();
  });

  it('should maintain reflect metadata', () => {
    expect(Reflect.getMetadata('some-metadata', instance.metadata)).toBe(true);
  });

  it('should preserve the original method name', () => {
    expect(instance.singleSpan.name).toBe('singleSpan');
  });

  it('should set correct span', () => {
    instance.singleSpan();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans.map((span) => span.name)).toEqual(['TestSpan.singleSpan']);
  });

  it('should set correct span options', () => {
    instance.fooProducerSpan();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans.map((span) => span.kind)).toEqual([SpanKind.PRODUCER]);
  });

  it('should set correct span options with implicit span name', () => {
    instance.implicitSpanNameWithOptions();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe('TestSpan.implicitSpanNameWithOptions');
    expect(spans[0].kind).toBe(SpanKind.PRODUCER);
  });

  it('should set correct span options based on method params', () => {
    instance.argsInOptions(10, 'bar');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes).toEqual({ a: 10, b: 'bar' });
  });

  it('should set correct span options based on method params with implicit span name', () => {
    instance.argsInOptionsWithImplicitName(10, 'bar');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes).toEqual({ a: 10, b: 'bar' });
  });

  it('should set correct span even when calling other method with Span decorator', () => {
    instance.doubleSpan();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(2);
    expect(spans.map((span) => span.name)).toEqual([
      'TestSpan.singleSpan',
      'TestSpan.doubleSpan',
    ]);
  });

  it('should propagate errors', () => {
    expect(() => instance.error()).toThrow('hello world');
  });

  it('should set setStatus to ERROR and message to error message', () => {
    expect(() => instance.error()).toThrow('hello world');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status).toEqual({
      code: SpanStatusCode.ERROR,
      message: 'hello world',
    });
  });

  it('should set recordException with error', () => {
    expect(() => instance.error()).toThrow('hello world');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].events).toHaveLength(1);
    const event = spans[0].events[0];
    expect(event.name).toBe('exception');
    expect(event.attributes).toBeDefined();
    expect(event.droppedAttributesCount).toBe(0);
    expect(event.time).toBeDefined();
  });

  it('should handle symbols', () => {
    instance[symbol]();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans.map((span) => span.name)).toEqual(['TestSpan.Symbol(testSymbol)']);
  });

  it('should set attributes from onResult in sync method', () => {
    const result = instance.syncMethod();
    expect(result).toBe('success');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes).toEqual({ result: 'success' });
  });

  it('should set attributes from onResult in async method', async () => {
    const result = await instance.asyncMethod();
    expect(result).toBe('async success');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes).toEqual({ result: 'async success' });
  });

  it('should record exception if onResult throws', () => {
    const result = instance.errorInOnResult();
    expect(result).toBe('success');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(spans[0].status.message).toBe('onResult error');
    expect(spans[0].events).toHaveLength(1);
    expect(spans[0].events[0].name).toBe('exception');
  });

  it('should call onResult with the thenable (not resolved value) for lazy thenables', () => {
    // The decorator hands the lazy thenable to onResult untouched (it cannot
    // know it is a deferred query). Accessing fields on the resolved shape
    // (e.g. `.rows`) therefore throws synchronously, and the decorator records
    // that as an exception on the span.
    const returned = instance.lazyThenableOnResult();
    expect(returned.triggered).toBe(false);

    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(spans[0].status.message).toMatch(
      "Cannot read properties of undefined (reading 'length')",
    );
    expect(spans[0].events).toHaveLength(1);
    expect(spans[0].events[0].name).toBe('exception');
    expect(spans[0].events[0].attributes?.['exception.type']).toBe('TypeError');
  });

  it('should not trigger a lazy thenable returned by the wrapped method', () => {
    const returned = instance.returnsLazyThenable();
    // The decorator must hand the lazy thenable back to the caller untouched.
    // It must NOT subscribe via .then(), which would force-execute query
    // builders such as Knex, Mongoose Query, or Drizzle queries that the
    // caller intended to defer (or compose further) before awaiting.
    expect(instance.lastLazyThenable?.triggered).toBe(false);
    expect(returned).toBe(instance.lastLazyThenable);
  });

  it('should end the span synchronously before the caller awaits a lazy thenable', () => {
    instance.returnsLazyThenable();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe('TestSpan.returnsLazyThenable');
  });

  it('should not record errors from a lazy thenable that rejects after the span has ended', async () => {
    const returned = instance.returnsFailingLazyThenable();
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.UNSET);
    expect(spans[0].events).toHaveLength(0);

    await expect(Promise.resolve(returned)).rejects.toThrow('lazy rejection');

    expect(traceExporter.getFinishedSpans()[0].events).toHaveLength(0);
  });

  it('should still track results from async (real Promise) methods', async () => {
    const result = await instance.asyncMethod();
    expect(result).toBe('async success');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe('TestSpan.asyncMethod');
    expect(spans[0].status.code).toBe(SpanStatusCode.UNSET);
    expect(spans[0].attributes).toEqual({ result: 'async success' });
  });

  it('should still track errors thrown by async (real Promise) methods', async () => {
    await expect(instance.asyncError()).rejects.toThrow('async hello world');
    const spans = traceExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe('TestSpan.asyncError');
    expect(spans[0].status).toEqual({
      code: SpanStatusCode.ERROR,
      message: 'async hello world',
    });
    expect(spans[0].events).toHaveLength(1);
    expect(spans[0].events[0].name).toBe('exception');
  });

  describe('asLangfuseTrace', () => {
    it('sets langfuse.trace.name to the implicit Class.method name', () => {
      instance.langfuseTraceImplicitName();
      const spans = traceExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].attributes['langfuse.trace.name']).toBe(
        'TestSpan.langfuseTraceImplicitName',
      );
    });

    it('sets langfuse.trace.name to an explicitly-provided span name', () => {
      instance.langfuseTraceExplicitName();
      const spans = traceExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].attributes['langfuse.trace.name']).toBe('explicitTraceName');
    });

    it('does not set langfuse.trace.name when the flag is absent', () => {
      instance.notLangfuseTrace();
      const spans = traceExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].attributes['langfuse.trace.name']).toBeUndefined();
    });
  });

  describe('observationKind', () => {
    it("translates observationKind: 'retriever' to langfuse.observation.type", () => {
      instance.retrieverKind();
      const spans = traceExporter.getFinishedSpans();
      expect(spans[0].attributes['langfuse.observation.type']).toBe('retriever');
    });

    it("translates observationKind: 'embedding' to langfuse.observation.type", () => {
      instance.embeddingKind();
      const spans = traceExporter.getFinishedSpans();
      expect(spans[0].attributes['langfuse.observation.type']).toBe('embedding');
    });

    it("does not emit the attribute for observationKind: 'span'", () => {
      instance.spanKind();
      const spans = traceExporter.getFinishedSpans();
      expect(spans[0].attributes['langfuse.observation.type']).toBeUndefined();
    });

    it('does not emit the attribute when observationKind is omitted', () => {
      instance.singleSpan();
      const spans = traceExporter.getFinishedSpans();
      expect(spans[0].attributes['langfuse.observation.type']).toBeUndefined();
    });
  });

  describe('captureIo', () => {
    let initialEnabled: boolean;

    beforeAll(() => {
      initialEnabled = isLangfuseEnabled();
    });

    afterAll(() => {
      setLangfuseEnabled(initialEnabled);
    });

    describe('when Langfuse is enabled', () => {
      beforeEach(() => {
        setLangfuseEnabled(true);
      });

      it('captures input as a JSON-stringified args array', () => {
        instance.capturesIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBe(
          JSON.stringify([{ query: 'hello' }]),
        );
      });

      it('captures the return value as output', () => {
        instance.capturesIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.output']).toBe(
          JSON.stringify({ rows: [1, 2, 3] }),
        );
      });

      it('captures input from async methods and output once the promise resolves', async () => {
        await instance.capturesIoAsync({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBe(
          JSON.stringify([{ query: 'hello' }]),
        );
        expect(spans[0].attributes['langfuse.observation.output']).toBe(
          JSON.stringify({ rows: [5] }),
        );
      });

      it('captures input even when the method throws; output is skipped', () => {
        expect(() => instance.capturesIoThrows({ query: 'hello' })).toThrow('boom');
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBe(
          JSON.stringify([{ query: 'hello' }]),
        );
        expect(spans[0].attributes['langfuse.observation.output']).toBeUndefined();
      });

      it('captures neither when captureIo is false', () => {
        instance.capturesIoOptOut({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBeUndefined();
        expect(spans[0].attributes['langfuse.observation.output']).toBeUndefined();
      });

      it('mirrors observation IO to langfuse.trace.input/output when asLangfuseTrace is set', () => {
        instance.langfuseTraceCapturesIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        const expectedInput = JSON.stringify([{ query: 'hello' }]);
        const expectedOutput = JSON.stringify({ rows: [1, 2, 3] });
        expect(spans[0].attributes['langfuse.observation.input']).toBe(expectedInput);
        expect(spans[0].attributes['langfuse.observation.output']).toBe(expectedOutput);
        expect(spans[0].attributes['langfuse.trace.input']).toBe(expectedInput);
        expect(spans[0].attributes['langfuse.trace.output']).toBe(expectedOutput);
      });

      it('does not set trace IO when captureIo is opted out, even with asLangfuseTrace', () => {
        instance.langfuseTraceOptedOutOfIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.trace.input']).toBeUndefined();
        expect(spans[0].attributes['langfuse.trace.output']).toBeUndefined();
      });

      it('does not set trace IO on non-trace spans, even when capturing observation IO', () => {
        instance.capturesIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBeDefined();
        expect(spans[0].attributes['langfuse.trace.input']).toBeUndefined();
        expect(spans[0].attributes['langfuse.trace.output']).toBeUndefined();
      });
    });

    describe('when Langfuse is disabled', () => {
      beforeEach(() => {
        setLangfuseEnabled(false);
      });

      it('captures nothing - even with captureIo defaulted on', () => {
        instance.capturesIo({ query: 'hello' });
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBeUndefined();
        expect(spans[0].attributes['langfuse.observation.output']).toBeUndefined();
      });

      it('captures nothing even on throw', () => {
        expect(() => instance.capturesIoThrows({ query: 'hello' })).toThrow('boom');
        const spans = traceExporter.getFinishedSpans();
        expect(spans[0].attributes['langfuse.observation.input']).toBeUndefined();
      });
    });
  });
});
