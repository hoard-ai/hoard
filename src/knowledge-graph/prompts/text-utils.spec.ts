import { KgNodeFactory } from '@/test/factories';

import {
  concatenateEpisodes,
  formatPreviousEpisodes,
  selectChunkText,
  truncateAtSentence,
} from './text-utils';

describe('truncateAtSentence', () => {
  it('returns the text unchanged when it is already within maxChars', () => {
    const text = 'Short sentence.';
    expect(truncateAtSentence(text, 100)).toBe(text);
  });

  it('truncates at the last complete sentence before maxChars', () => {
    const text = 'First sentence. Second sentence. Third sentence.';
    // maxChars = 35 falls inside "Third sentence." - should cut after "Second sentence."
    expect(truncateAtSentence(text, 35)).toBe('First sentence. Second sentence.');
  });

  it('falls back to a hard character truncate when no sentence boundary fits', () => {
    const text = 'noPunctuationAtAllJustAReallyLongStringOfText';
    const out = truncateAtSentence(text, 10);
    expect(out).toBe('noPunctuat');
    expect(out.length).toBe(10);
  });

  it('does not split on common single-word abbreviations (Dr., Mr., etc.)', () => {
    // Without the abbreviation guard, "Dr." would be treated as a boundary
    // and we would truncate to "Dr." losing the rest of the sentence.
    const text = 'Dr. Osei presented results. The trial ran for six months.';
    expect(truncateAtSentence(text, 30)).toBe('Dr. Osei presented results.');

    const text2 = 'See Mr. Smith later. The meeting starts soon.';
    expect(truncateAtSentence(text2, 22)).toBe('See Mr. Smith later.');
  });

  it('does not split on decimal numbers', () => {
    const text = 'The value of pi is 3.14 approximately. Use it carefully.';
    // The "." inside "3.14" must NOT be treated as a sentence boundary.
    expect(truncateAtSentence(text, 45)).toBe('The value of pi is 3.14 approximately.');
  });

  it('treats ! and ? as sentence endings', () => {
    const text = 'Wow! Is this real? Tell me more about it.';
    expect(truncateAtSentence(text, 20)).toBe('Wow! Is this real?');
  });
});

describe('concatenateEpisodes', () => {
  it('formats each episode with [Episode N] (iso)\\n<content> and joins with blank line', () => {
    const e0 = KgNodeFactory.createEpisodicNode({
      name: 'E0',
      content: 'first content',
      validAt: new Date('2026-01-01T00:00:00.000Z'),
    });
    const e1 = KgNodeFactory.createEpisodicNode({
      name: 'E1',
      content: 'second content',
      validAt: new Date('2026-02-01T00:00:00.000Z'),
    });

    expect(concatenateEpisodes([e0, e1])).toBe(
      '[Episode 0] (2026-01-01T00:00:00Z)\nfirst content\n\n' +
        '[Episode 1] (2026-02-01T00:00:00Z)\nsecond content',
    );
  });
});

describe('formatPreviousEpisodes', () => {
  it('returns "None" for an empty list', () => {
    expect(formatPreviousEpisodes([])).toBe('None');
  });

  it('formats each episode as "- [name] (iso): content"', () => {
    const e0 = KgNodeFactory.createEpisodicNode({
      name: 'Prev',
      content: 'prior content',
      validAt: new Date('2025-12-31T00:00:00.000Z'),
    });
    expect(formatPreviousEpisodes([e0])).toBe(
      '- [Prev] (2025-12-31T00:00:00Z): prior content',
    );
  });
});

describe('selectChunkText', () => {
  it('passes a single-chunk episode through unchanged (no labels)', () => {
    expect(selectChunkText(new Set([0]), ['only chunk'])).toBe('only chunk');
  });

  it('labels a single excerpt of a multi-chunk episode (1-based)', () => {
    const out = selectChunkText(new Set([2]), ['a', 'b', 'c', 'd']);
    expect(out).toContain('[Chunk 3]\nc');
    expect(out).toContain('excerpt (one chunk)');
    expect(out).not.toContain('[Chunk 1]');
    expect(out).not.toContain('[Chunk 2]');
    expect(out).not.toContain('[Chunk 4]');
  });

  it('labels multiple excerpts in ascending order with the overlap header', () => {
    const out = selectChunkText(new Set([3, 1]), ['a', 'b', 'c', 'd']);
    expect(out).toContain('may overlap and need not be contiguous');
    expect(out.indexOf('[Chunk 2]')).toBeLessThan(out.indexOf('[Chunk 4]'));
    expect(out).toContain('[Chunk 2]\nb');
    expect(out).toContain('[Chunk 4]\nd');
  });

  it('falls back to the whole episode when indices are empty (unknown provenance)', () => {
    const out = selectChunkText(new Set(), ['a', 'b']);
    expect(out).toContain('[Chunk 1]\na');
    expect(out).toContain('[Chunk 2]\nb');
    expect(out).toContain('may overlap and need not be contiguous');
  });

  it('throws on an out-of-range chunk index (provenance bookkeeping bug)', () => {
    expect(() => selectChunkText(new Set([9]), ['a', 'b'])).toThrow();
    expect(() => selectChunkText(new Set([-1]), ['a', 'b'])).toThrow();
  });
});
