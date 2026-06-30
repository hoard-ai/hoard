/**
 * Best-effort JSON serialization for span attributes. OTel's `setAttribute`
 * rejects non-primitive values and `@langfuse/otel`'s `LangfuseSpanProcessor`
 * doesn't serialize for you (only `@langfuse/tracing`'s SDK helpers do, and
 * we go straight through OTel) - so we serialize before setting the attribute.
 */
const MAX_ARRAY_LENGTH = 64;

// Backstop cap on replacer invocations, so traversal aborts in bounded time on
// any unexpectedly large or self-generating structure.
const MAX_NODES = 50_000;

/**
 * Prisma client / interactive-transaction proxies mint fresh delegate objects on
 * every property access, so a `WeakSet` cycle guard never matches and
 * `JSON.stringify` recurses for seconds before throwing.
 */
function isPrismaClient(val: object): boolean {
  try {
    return '$executeRaw' in val || '$queryRaw' in val;
  } catch {
    return false;
  }
}

export function safeStringify(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    const seen = new WeakSet<object>();
    let nodes = 0;
    return JSON.stringify(value, (_key, val: unknown) => {
      if (++nodes > MAX_NODES) {
        throw new Error('safeStringify: node budget exceeded');
      }
      if (Array.isArray(val) && val.length >= MAX_ARRAY_LENGTH) {
        return `<oversized_array:${val.length}>`;
      }
      if (typeof val === 'object' && val !== null) {
        if (seen.has(val)) return '<cycle>';
        if (isPrismaClient(val)) return '<prisma client>';
        seen.add(val);
      }
      return val;
    });
  } catch {
    return '<failed to serialize>';
  }
}
