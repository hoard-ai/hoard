import { EpisodeType } from '@/knowledge-graph/models';

import {
  CHUNK_MAX_TOKENS,
  chunkJsonData,
  chunkMessageContent,
  chunkTextContent,
  countJsonKeys,
  estimateTokens,
  jsonLikelyDense,
  prepareChunks,
  textLikelyDense,
} from './content-chunking';

describe('prepareChunks', () => {
  it('returns single chunk for short text', () => {
    expect(prepareChunks('hi', EpisodeType.text)).toEqual(['hi']);
  });

  it('returns single chunk for short JSON', () => {
    const json = JSON.stringify([{ a: 1 }, { b: 2 }]);
    expect(prepareChunks(json, EpisodeType.json)).toEqual([json]);
  });

  it('splits large entity-dense JSON arrays', () => {
    // Each item ~17 chars => density ~4/17 = 0.24 > 0.15 threshold; 2000 items ~ 34000 chars
    // > 12000 char chunk size => multiple chunks
    const items = Array.from({ length: 2000 }, (_, i) => ({ id: i, v: 'x' }));
    const json = JSON.stringify(items);
    const result = prepareChunks(json, EpisodeType.json);
    expect(result.length).toBeGreaterThan(1);
    for (const chunk of result) {
      expect(() => JSON.parse(chunk) as unknown).not.toThrow();
    }
  });

  it('throws on invalid JSON above the min-tokens gate', () => {
    const notJson = 'not json at all ' + 'x'.repeat(20000);
    expect(() => prepareChunks(notJson, EpisodeType.json)).toThrow(SyntaxError);
  });

  it('returns single chunk for large but sparse JSON', () => {
    // Two elements, ~2500 chars each => ~5000 chars / ~1250 tokens (above CHUNK_MIN_TOKENS=1000).
    // Density = 2/1250 * 1000 = 1.6, far below the 150 threshold => no chunking.
    const data = [{ content: 'x'.repeat(2500) }, { content: 'y'.repeat(2500) }];
    const json = JSON.stringify(data);
    expect(prepareChunks(json, EpisodeType.json)).toEqual([json]);
  });

  it('returns single chunk for large low-density prose', () => {
    const prose =
      'the sun was setting over the horizon as the old man walked slowly ' +
      'down the dusty road. he had been traveling for many days and his ' +
      'feet were tired. the journey had been long but he knew that soon ' +
      'he would reach his destination. the wind whispered through the trees ' +
      'and the birds sang their evening songs. ';
    // ~280 chars per repeat * 20 = ~5600 chars / ~1400 tokens, above CHUNK_MIN_TOKENS=1000.
    // No mid-sentence capitalized words => density ~0, far below threshold.
    const content = prose.repeat(20);
    expect(prepareChunks(content, EpisodeType.text)).toEqual([content]);
  });

  it('force-chunks low-density prose at/above the max-token ceiling', () => {
    const prose =
      'the sun was setting over the horizon as the old man walked slowly ' +
      'down the dusty road. he had been traveling for many days and his ' +
      'feet were tired. the journey had been long but he knew that soon ' +
      'he would reach his destination. the wind whispered through the trees ' +
      'and the birds sang their evening songs. ';
    // Grow past the ceiling. Density stays ~0 (no mid-sentence capitals), so a
    // split here can only come from the max-token override, not the density gate.
    const content = prose.repeat(Math.ceil((CHUNK_MAX_TOKENS * 4 * 1.5) / prose.length));
    const tokens = estimateTokens(content);
    expect(tokens).toBeGreaterThanOrEqual(CHUNK_MAX_TOKENS);
    expect(textLikelyDense(content, tokens)).toBe(false);
    expect(prepareChunks(content, EpisodeType.text).length).toBeGreaterThan(1);
  });
});

describe('chunkJsonData', () => {
  describe('arrays', () => {
    it('does not chunk a small array', () => {
      const data = [{ name: 'Alice' }, { name: 'Bob' }];
      const chunks = chunkJsonData(data, 1000);
      expect(chunks).toHaveLength(1);
      expect(JSON.parse(chunks[0])).toEqual(data);
    });

    it('returns "[]" for an empty array', () => {
      expect(chunkJsonData([], 100)).toEqual(['[]']);
    });

    it('splits at element boundaries; every chunk is a valid JSON array of complete elements', () => {
      const data = Array.from({ length: 20 }, (_, i) => ({
        id: i,
        data: 'x'.repeat(100),
      }));
      const chunks = chunkJsonData(data, 100, 20);
      expect(chunks.length).toBeGreaterThan(1);
      for (const chunk of chunks) {
        const parsed = JSON.parse(chunk) as Array<{ id: number; data: string }>;
        expect(Array.isArray(parsed)).toBe(true);
        for (const item of parsed) {
          expect(item).toHaveProperty('id');
          expect(item).toHaveProperty('data');
        }
      }
    });

    it('preserves every element across chunks', () => {
      const data = Array.from({ length: 10 }, (_, i) => ({ id: i }));
      const chunks = chunkJsonData(data, 50, 10);
      const seen = new Set<number>();
      for (const chunk of chunks) {
        const parsed = JSON.parse(chunk) as Array<{ id: number }>;
        for (const item of parsed) seen.add(item.id);
      }
      expect(seen).toEqual(new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
    });
  });

  describe('objects', () => {
    it('does not chunk a small object', () => {
      const data = { name: 'Alice', age: 30 };
      const chunks = chunkJsonData(data, 1000);
      expect(chunks).toHaveLength(1);
      expect(JSON.parse(chunks[0])).toEqual(data);
    });

    it('returns "{}" for an empty object', () => {
      expect(chunkJsonData({}, 100)).toEqual(['{}']);
    });

    it('splits at key boundaries; every chunk is a valid JSON object of complete entries', () => {
      const data: Record<string, string> = {};
      for (let i = 0; i < 20; i++) data[`key_${i}`] = 'x'.repeat(100);
      const chunks = chunkJsonData(data, 100, 20);
      expect(chunks.length).toBeGreaterThan(1);
      for (const chunk of chunks) {
        const parsed = JSON.parse(chunk) as Record<string, string>;
        expect(typeof parsed).toBe('object');
        expect(Array.isArray(parsed)).toBe(false);
        for (const key of Object.keys(parsed)) {
          expect(key).toMatch(/^key_/);
        }
      }
    });

    it('preserves every key across chunks', () => {
      const data: Record<string, string> = {};
      for (let i = 0; i < 10; i++) data[`key_${i}`] = `value_${i}`;
      const chunks = chunkJsonData(data, 50, 10);
      const seen = new Set<string>();
      for (const chunk of chunks) {
        const parsed = JSON.parse(chunk) as Record<string, string>;
        for (const k of Object.keys(parsed)) seen.add(k);
      }
      const expected = new Set(Array.from({ length: 10 }, (_, i) => `key_${i}`));
      expect(seen).toEqual(expected);
    });
  });

  describe('overlap', () => {
    it('shares at least one element between every pair of adjacent array chunks', () => {
      const data = Array.from({ length: 30 }, (_, i) => ({
        id: i,
        name: `Entity ${i}`,
      }));
      const chunks = chunkJsonData(data, 80, 30);
      expect(chunks.length).toBeGreaterThan(1);
      for (let i = 0; i < chunks.length - 1; i++) {
        const curIds = (JSON.parse(chunks[i]) as Array<{ id: number }>).map((x) => x.id);
        const nxtIds = new Set(
          (JSON.parse(chunks[i + 1]) as Array<{ id: number }>).map((x) => x.id),
        );
        const shared = curIds.filter((id) => nxtIds.has(id));
        expect(shared.length).toBeGreaterThan(0);
      }
    });
  });

  describe('edge cases', () => {
    it('returns at least one chunk for an oversized single element', () => {
      const data = [{ content: 'x'.repeat(10000) }];
      const chunks = chunkJsonData(data, 100, 10);
      expect(chunks.length).toBeGreaterThanOrEqual(1);
    });
  });
});

describe('chunkTextContent', () => {
  it('does not chunk short text', () => {
    const text = 'This is a short text.';
    const chunks = chunkTextContent(text, 1000);
    expect(chunks).toEqual([text]);
  });

  it('splits at paragraph boundaries without trailing-space chunks', () => {
    const paragraphs = ['Paragraph one.', 'Paragraph two.', 'Paragraph three.'];
    const text = paragraphs.join('\n\n');
    const chunks = chunkTextContent(text, 10, 5);
    for (const chunk of chunks) {
      expect(chunk.endsWith(' ')).toBe(false);
    }
  });

  it('splits a single large paragraph at sentence boundaries', () => {
    const sentences = Array.from(
      { length: 20 },
      (_, i) => `This is sentence number ${i}.`,
    );
    const longParagraph = sentences.join(' ');
    const chunks = chunkTextContent(longParagraph, 50, 10);
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('preserves every word across chunks', () => {
    const text = 'Alpha beta gamma delta epsilon zeta eta theta.';
    const chunks = chunkTextContent(text, 10, 2);
    const allWords = new Set(text.replace(/\./g, '').split(/\s+/));
    const found = new Set<string>();
    for (const chunk of chunks) {
      for (const w of chunk.replace(/\./g, '').split(/\s+/)) {
        if (w) found.add(w);
      }
    }
    for (const w of allWords) expect(found.has(w)).toBe(true);
  });

  it('shares at least one word between every pair of adjacent chunks', () => {
    const paragraphs = Array.from(
      { length: 10 },
      (_, i) => `Paragraph ${i} with some content here.`,
    );
    const text = paragraphs.join('\n\n');
    const chunks = chunkTextContent(text, 50, 20);
    expect(chunks.length).toBeGreaterThan(1);
    for (let i = 0; i < chunks.length - 1; i++) {
      const curWords = new Set(chunks[i].split(/\s+/).filter(Boolean));
      const nxtWords = new Set(chunks[i + 1].split(/\s+/).filter(Boolean));
      const shared = [...curWords].filter((w) => nxtWords.has(w));
      expect(shared.length).toBeGreaterThan(0);
    }
  });
});

describe('chunkMessageContent', () => {
  it('does not chunk short message content', () => {
    const content = 'Alice: Hello!\nBob: Hi there!';
    const chunks = chunkMessageContent(content, 1000);
    expect(chunks).toEqual([content]);
  });

  it('preserves speaker:message format on every non-empty line of every chunk', () => {
    const messages = Array.from(
      { length: 10 },
      (_, i) => `Speaker${i}: This is message number ${i}.`,
    );
    const content = messages.join('\n');
    const chunks = chunkMessageContent(content, 50, 10);
    for (const chunk of chunks) {
      const lines = chunk.split('\n').filter((l) => l.trim().length > 0);
      for (const line of lines) {
        expect(line).toContain(':');
      }
    }
  });
});

describe('jsonLikelyDense', () => {
  it('detects dense arrays (many small elements)', () => {
    // 100 items, ~9 chars each => density ~444/1000 tokens, well above 150 threshold
    const data = Array.from({ length: 100 }, (_, i) => ({ id: i }));
    const tokens = estimateTokens(JSON.stringify(data));
    expect(jsonLikelyDense(data, tokens)).toBe(true);
  });

  it('does not mark sparse arrays as dense (few elements, lots of content)', () => {
    const data = [{ content: 'x'.repeat(1000) }, { content: 'y'.repeat(1000) }];
    const tokens = estimateTokens(JSON.stringify(data));
    expect(jsonLikelyDense(data, tokens)).toBe(false);
  });

  it('detects dense objects (many keys)', () => {
    const data: Record<string, string> = {};
    for (let i = 0; i < 50; i++) data[`k${i}`] = `v${i}`;
    const tokens = estimateTokens(JSON.stringify(data));
    expect(jsonLikelyDense(data, tokens)).toBe(true);
  });

  it('returns false when tokens is 0', () => {
    expect(jsonLikelyDense([1, 2, 3], 0)).toBe(false);
  });
});

describe('textLikelyDense', () => {
  it('detects entity-rich text (many mid-sentence capitals)', () => {
    let text = 'Alice met Bob at Acme Corp. Then Carol and David joined them. ';
    text += 'Eve from Globex introduced Frank and Grace. ';
    text += 'Later Henry and Iris arrived from Initech. ';
    text = text.repeat(10);
    expect(textLikelyDense(text, estimateTokens(text))).toBe(true);
  });

  it('does not mark narrative prose as dense', () => {
    const prose = (
      'the sun was setting over the horizon as the old man walked slowly ' +
      'down the dusty road. he had been traveling for many days and his ' +
      'feet were tired. the journey had been long but he knew that soon ' +
      'he would reach his destination. the wind whispered through the trees ' +
      'and the birds sang their evening songs. '
    ).repeat(10);
    expect(textLikelyDense(prose, estimateTokens(prose))).toBe(false);
  });

  it('ignores capitals that are sentence starters', () => {
    const text = 'This is a sentence. Another one follows. Yet another here. '.repeat(50);
    expect(textLikelyDense(text, estimateTokens(text))).toBe(false);
  });
});

describe('countJsonKeys', () => {
  it('counts shallow nested keys up to the depth budget', () => {
    const data = {
      a: 1,
      b: { c: 2, d: 3 },
      e: [{ f: 4 }, { g: 5 }],
    };
    // depth 2: a, b, c, d, e, f, g = 7
    expect(countJsonKeys(data, 2)).toBe(7);
  });

  it('respects the depth limit', () => {
    const data = { a: { b: { c: { d: 1 } } } };
    expect(countJsonKeys(data, 1)).toBe(1);
    expect(countJsonKeys(data, 2)).toBe(2);
  });
});
