import { safeStringify } from './serialize';

describe('safeStringify', () => {
  describe('primitives', () => {
    it('serializes null', () => {
      expect(safeStringify(null)).toBe('null');
    });

    it('serializes numbers and booleans via JSON.stringify', () => {
      expect(safeStringify(42)).toBe('42');
      expect(safeStringify(true)).toBe('true');
    });

    it('passes strings through unchanged so pre-serialized payloads are not double-quoted', () => {
      expect(safeStringify('already a string')).toBe('already a string');
    });
  });

  describe('objects and arrays', () => {
    it('round-trips a plain object', () => {
      expect(safeStringify({ a: 1, b: 'hello' })).toBe('{"a":1,"b":"hello"}');
    });

    it('round-trips a mixed array', () => {
      expect(safeStringify([1, 'two', { three: 3 }])).toBe('[1,"two",{"three":3}]');
    });

    it('serializes Date as ISO string (delegates to Date.prototype.toJSON)', () => {
      const iso = '2026-05-27T00:00:00.000Z';
      expect(safeStringify(new Date(iso))).toBe(`"${iso}"`);
    });
  });

  describe('oversized array threshold', () => {
    it('keeps an array just below the threshold inline', () => {
      const arr = Array.from({ length: 63 }, (_, i) => i);
      expect(safeStringify({ arr })).toBe(`{"arr":${JSON.stringify(arr)}}`);
    });

    it('collapses an array at the threshold to <oversized_array:N>', () => {
      const arr = Array.from({ length: 64 }, (_, i) => i);
      expect(safeStringify({ arr })).toBe('{"arr":"<oversized_array:64>"}');
    });

    it('reports the real length for embedding-sized arrays', () => {
      const embedding = Array.from({ length: 768 }, () => 0.5);
      expect(safeStringify({ embedding })).toBe('{"embedding":"<oversized_array:768>"}');
    });

    it('collapses oversized arrays nested under each entry of a parent array', () => {
      // Real-world shape: list of nodes each carrying an embedding vector.
      const nodes = [
        { id: 'a', nameEmbedding: Array.from({ length: 768 }, () => 0.1) },
        { id: 'b', nameEmbedding: Array.from({ length: 768 }, () => 0.2) },
      ];
      expect(safeStringify(nodes)).toBe(
        '[{"id":"a","nameEmbedding":"<oversized_array:768>"},' +
          '{"id":"b","nameEmbedding":"<oversized_array:768>"}]',
      );
    });
  });

  describe('Maps and Sets', () => {
    it('renders a string-keyed Map as a plain object', () => {
      const map = new Map([
        ['a', 1],
        ['b', 2],
      ]);
      expect(safeStringify({ map })).toBe('{"map":{"a":1,"b":2}}');
    });

    it('renders a non-string-keyed Map as entry pairs', () => {
      const map = new Map([[['Entity', 'Entity'], ['WORKS_AT']]]);
      expect(safeStringify(map)).toBe('[[["Entity","Entity"],["WORKS_AT"]]]');
    });

    it('renders a Set as an array', () => {
      expect(safeStringify(new Set(['x', 'y']))).toBe('["x","y"]');
    });

    it('collapses a Map at the array threshold to <oversized_map:N>', () => {
      const map = new Map(Array.from({ length: 64 }, (_, i) => [`k${i}`, i]));
      expect(safeStringify({ map })).toBe('{"map":"<oversized_map:64>"}');
    });

    it('collapses a Set at the array threshold to <oversized_set:N>', () => {
      const set = new Set(Array.from({ length: 64 }, (_, i) => i));
      expect(safeStringify({ set })).toBe('{"set":"<oversized_set:64>"}');
    });

    it('applies nested rules inside converted Map values (embedding arrays still collapse)', () => {
      const map = new Map([
        ['node', { nameEmbedding: Array.from({ length: 768 }, () => 0.1) }],
      ]);
      expect(safeStringify(map)).toBe(
        '{"node":{"nameEmbedding":"<oversized_array:768>"}}',
      );
    });

    it('guards cycles through a Map', () => {
      const map = new Map<string, unknown>();
      map.set('self', map);
      expect(safeStringify(map)).toBe('{"self":"<cycle>"}');
    });
  });

  describe('cycles', () => {
    it('replaces direct self-cycles with <cycle> instead of losing the whole payload', () => {
      type Cyclic = { a: number; self?: Cyclic };
      const obj: Cyclic = { a: 1 };
      obj.self = obj;
      expect(safeStringify(obj)).toBe('{"a":1,"self":"<cycle>"}');
    });

    it('flags shared (non-cyclic) refs as <cycle> after the first occurrence', () => {
      // Accepted false positive: the WeakSet does not track parent chains, so a
      // diamond reference looks identical to a cycle. Fine for span attributes -
      // we get one good serialization; later refs just show <cycle>.
      const shared = { x: 1 };
      expect(safeStringify({ a: shared, b: shared })).toBe('{"a":{"x":1},"b":"<cycle>"}');
    });
  });

  describe('failure fallback', () => {
    it('returns <failed to serialize> when toJSON throws', () => {
      const trap = {
        toJSON(): never {
          throw new Error('nope');
        },
      };
      expect(safeStringify(trap)).toBe('<failed to serialize>');
    });

    it('returns <failed to serialize> for a BigInt at the top level', () => {
      expect(safeStringify(BigInt(10))).toBe('<failed to serialize>');
    });

    it('returns <failed to serialize> for a BigInt nested in an object', () => {
      expect(safeStringify({ count: 10n })).toBe('<failed to serialize>');
    });
  });

  describe('JSON.stringify native behavior (pinned, not redefined)', () => {
    it('drops undefined / function / symbol-valued properties from objects', () => {
      expect(
        safeStringify({
          keep: 1,
          dropUndefined: undefined,
          dropFn: () => 0,
          dropSymbolValue: Symbol('x'),
        }),
      ).toBe('{"keep":1}');
    });

    it('emits null in arrays for undefined / function / symbol elements', () => {
      expect(safeStringify([1, undefined, () => 0, Symbol('x'), 2])).toBe(
        '[1,null,null,null,2]',
      );
    });
  });
});
