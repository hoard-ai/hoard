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

function serializeMap(map: Map<unknown, unknown>): unknown {
  if (map.size >= MAX_ARRAY_LENGTH) return `<oversized_map:${map.size}>`;
  const entries = [...map.entries()];
  if (
    entries.every((entry): entry is [string, unknown] => typeof entry[0] === 'string')
  ) {
    return Object.fromEntries(entries);
  }
  return entries;
}

function serializeSet(set: Set<unknown>): unknown {
  if (set.size >= MAX_ARRAY_LENGTH) return `<oversized_set:${set.size}>`;
  return [...set];
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
        if (val instanceof Map) return serializeMap(val);
        if (val instanceof Set) return serializeSet(val);
      }
      return val;
    });
  } catch {
    return '<failed to serialize>';
  }
}
