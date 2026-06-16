import type { EpisodicNode } from '../models';

export const MAX_SUMMARY_CHARS = 1000;

// Node's Date.toISOString() emits millisecond precision ("2025-04-30T00:00:00.000Z"),
// but every prompt rule example uses the second-precision form ("2025-04-30T00:00:00Z").
// Strip milliseconds so the timestamps the model sees on input match the format it is
// instructed to emit on output.
export function formatPromptTimestamp(date: Date): string {
  return date.toISOString().replace(/\.\d{3}Z$/, 'Z');
}

const ABBREVIATIONS = new Set([
  // Latin shorthand
  'e.g',
  'i.e',
  'etc',
  'vs',
  // Personal titles
  'mr',
  'mrs',
  'dr',
  'prof',
  'sr',
  'jr',
  'rev',
  'hon',
  'capt',
  'sgt',
  'lt',
  'maj',
  'pres',
  'gov',
  'sen',
  'fr',
  'esq',
  // Organizational suffixes
  'inc',
  'corp',
  'ltd',
  'co',
  'bros',
  'dept',
  'univ',
  // Geographic
  'ave',
  'blvd',
  'rd',
  'mt',
  // Months
  'jan',
  'feb',
  'mar',
  'apr',
  'jun',
  'jul',
  'aug',
  'sep',
  'sept',
  'oct',
  'nov',
  'dec',
]);

export function isSentenceEnd(text: string, pos: number): boolean {
  const ch = text[pos];
  if (ch !== '.' && ch !== '!' && ch !== '?') return false;

  // Decimal numbers: digit.digit
  if (ch === '.' && pos > 0 && pos + 1 < text.length) {
    if (/\d/.test(text[pos - 1]) && /\d/.test(text[pos + 1])) return false;
  }

  // Abbreviation check: word ending with .
  if (ch === '.') {
    let start = pos - 1;
    while (start >= 0 && /[a-zA-Z]/.test(text[start])) start--;
    const word = text.slice(start + 1, pos).toLowerCase();
    if (ABBREVIATIONS.has(word)) return false;
  }

  // Must be followed by whitespace or end-of-string
  if (pos + 1 < text.length && !/\s/.test(text[pos + 1])) return false;

  return true;
}

export function truncateAtSentence(text: string, maxChars = MAX_SUMMARY_CHARS): string {
  if (text.length <= maxChars) return text;

  let lastBoundary = -1;
  for (let i = 0; i < maxChars; i++) {
    if (isSentenceEnd(text, i)) lastBoundary = i + 1;
  }

  if (lastBoundary > 0) return text.slice(0, lastBoundary).trimEnd();
  return text.slice(0, maxChars).trimEnd();
}

export function concatenateEpisodes(episodes: EpisodicNode[]): string {
  return episodes
    .map(
      (ep, i) => `[Episode ${i}] (${formatPromptTimestamp(ep.validAt)})\n${ep.content}`,
    )
    .join('\n\n');
}

export function formatPreviousEpisodes(episodes: EpisodicNode[]): string {
  if (episodes.length === 0) return 'None';
  return episodes
    .map((e) => `- [${e.name}] (${formatPromptTimestamp(e.validAt)}): ${e.content}`)
    .join('\n');
}

// TODO: Consider this: You feed a book that has Alice in 20 chunks.
// These 20 chunks are passed down to a prompt. Far too much. Needs a cap probably.

/**
 * Returns the episode text scoped to `chunkIndices` - the chunks an item came
 * from - for a per-item operation (summary, attributes, dedup, resolution). Feed
 * the result as `content` of a synthetic episode to `formatCurrentEpisode`.
 * A single-chunk episode passes its content through unchanged.
 *
 * An empty `chunkIndices` means "chunks unknown" and falls back to the whole
 * episode (all chunks) rather than yielding blank content. An out-of-range index
 * is a provenance-bookkeeping bug and throws rather than being silently dropped.
 */
export function selectChunkText(chunkIndices: Set<number>, allChunks: string[]): string {
  for (const i of chunkIndices) {
    if (i < 0 || i >= allChunks.length) {
      throw new Error(
        `selectChunkText: chunk index ${i} out of range (episode has ${allChunks.length} chunks)`,
      );
    }
  }
  if (allChunks.length <= 1) return allChunks[0] ?? '';

  // No known chunks for this item -> use the whole episode, never blank content.
  const sorted = [...chunkIndices].sort((a, b) => a - b);
  const effective = sorted.length > 0 ? sorted : allChunks.map((_, i) => i);

  const labeled = effective.map((i) => `[Chunk ${i + 1}]\n${allChunks[i]}`).join('\n\n');
  const header =
    effective.length === 1
      ? 'The following is an excerpt (one chunk) from a larger episode; surrounding context is omitted.'
      : 'The following are excerpts (chunks) from a larger episode. Chunks may overlap and need not be contiguous.';
  return `${header}\n\n${labeled}`;
}

export function formatCurrentEpisode(
  episode: EpisodicNode,
  opts: { includeSource?: boolean } = {},
): string {
  const lines = [
    `Name: ${episode.name}`,
    `Timestamp: ${formatPromptTimestamp(episode.validAt)}`,
  ];
  if (opts.includeSource) lines.push(`Source: ${episode.source}`);
  lines.push(`Content: ${episode.content}`);
  return lines.join('\n');
}
