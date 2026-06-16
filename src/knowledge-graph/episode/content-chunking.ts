import { EpisodeType } from '@/knowledge-graph/types';

import { isSentenceEnd } from '../prompts/text-utils';

// Density-based chunking: only chunk high-density content (many entities per token)
// This targets the failure case (large entity-dense inputs) while preserving
// context for prose/narrative content
export const CHUNK_TOKEN_SIZE = 3000;
export const CHUNK_OVERLAP_TOKENS = 200;
// Minimum tokens before considering chunking - short content processes fine regardless of density
export const CHUNK_MIN_TOKENS = 1000;
// Entity density threshold: chunk if estimated density > this value
// For JSON: elements per 1000 tokens > threshold * 1000 (e.g., 0.15 = 150 elements/1000 tokens)
// For Text: capitalized words per 1000 tokens > threshold * 500 (e.g., 0.15 = 75 caps/1000 tokens)
// Higher values = more conservative (less chunking), targets P95+ density cases
// Examples that trigger chunking at 0.15: AWS cost data (12mo), bulk data imports, entity-dense JSON
// Examples that DON'T chunk at 0.15: meeting transcripts, news articles, documentation
export const CHUNK_DENSITY_THRESHOLD = 0.15;
// Hard upper bound: content at/above this many tokens is ALWAYS chunked, regardless
// of density. Diverges from Graphiti (which has no ceiling, so large low-density
// prose is never split and floods extraction). Tune against the eval set.
export const CHUNK_MAX_TOKENS = 6000;

// Approximate characters per token (conservative estimate).
const CHARS_PER_TOKEN = 4;

export function estimateTokens(text: string): number {
  return Math.floor(text.length / CHARS_PER_TOKEN);
}

function tokensToChars(tokens: number): number {
  return tokens * CHARS_PER_TOKEN;
}

/* Overrides for the density gates + chunk sizing. Each field defaults to the
 * corresponding module constant, so omitting opts preserves prior behavior. */
export type ChunkOptions = {
  minTokens: number;
  maxTokens: number;
  densityThreshold: number;
  chunkTokenSize: number;
  overlapTokens: number;
};

/** Returns [content] when chunking isn't warranted (small or low-density); otherwise returns
 *  type-specific chunks. Always non-empty so callers can iterate uniformly. */
export function prepareChunks(
  content: string,
  source: EpisodeType,
  opts?: Partial<ChunkOptions>,
): string[] {
  const minTokens = opts?.minTokens ?? CHUNK_MIN_TOKENS;
  const maxTokens = opts?.maxTokens ?? CHUNK_MAX_TOKENS;
  const densityThreshold = opts?.densityThreshold ?? CHUNK_DENSITY_THRESHOLD;
  const chunkTokenSize = opts?.chunkTokenSize ?? CHUNK_TOKEN_SIZE;
  const overlapTokens = opts?.overlapTokens ?? CHUNK_OVERLAP_TOKENS;

  const tokens = estimateTokens(content);
  if (tokens < minTokens) return [content];

  // Above the ceiling, always chunk regardless of density (protects extraction
  // from oversized low-density inputs the density gate would otherwise pass through).
  const forceChunk = tokens >= maxTokens;

  if (source === EpisodeType.json) {
    // Schema refine on JsonEpisodeInputSchema guarantees parseable JSON.
    const data: unknown = JSON.parse(content);
    return forceChunk || jsonLikelyDense(data, tokens, densityThreshold)
      ? chunkJsonData(data, chunkTokenSize, overlapTokens)
      : [content];
  }

  if (!forceChunk && !textLikelyDense(content, tokens, densityThreshold))
    return [content];
  return source === EpisodeType.message
    ? chunkMessageContent(content, chunkTokenSize, overlapTokens)
    : chunkTextContent(content, chunkTokenSize, overlapTokens);
}

export function jsonLikelyDense(
  data: unknown,
  tokens: number,
  threshold = CHUNK_DENSITY_THRESHOLD,
): boolean {
  if (tokens === 0) return false;

  // Scalars (number/string/bool/null) yield count 0 -> density 0 -> below threshold.
  const count = Array.isArray(data)
    ? data.length
    : data !== null && typeof data === 'object'
      ? countJsonKeys(data, 2)
      : 0;

  const density = (count / tokens) * 1000;
  return density > threshold * 1000;
}

// Arrays are transparent to the depth budget - only object levels consume it.
export function countJsonKeys(data: unknown, maxDepth: number, currentDepth = 0): number {
  if (Array.isArray(data)) {
    let count = 0;
    for (const item of data) count += countJsonKeys(item, maxDepth, currentDepth);
    return count;
  }
  if (data === null || typeof data !== 'object') return 0;
  if (currentDepth >= maxDepth) return 0;

  let count = Object.keys(data).length;
  for (const value of Object.values(data)) {
    count += countJsonKeys(value, maxDepth, currentDepth + 1);
  }
  return count;
}

export function textLikelyDense(
  content: string,
  tokens: number,
  threshold = CHUNK_DENSITY_THRESHOLD,
): boolean {
  if (tokens === 0) return false;
  const words = content.match(/\S+/g) ?? [];
  if (words.length === 0) return false;

  let capitalizedCount = 0;
  for (let i = 0; i < words.length; i++) {
    if (i === 0) continue;
    const lastChar = words[i - 1].slice(-1);
    if (lastChar === '.' || lastChar === '!' || lastChar === '?') continue;

    // Check if capitalized (first char upper, not all caps)
    const cleaned = words[i].replace(/^[.,!?;:'"()[\]{}]+|[.,!?;:'"()[\]{}]+$/g, '');
    if (cleaned && /^[A-Z]/.test(cleaned) && cleaned !== cleaned.toUpperCase()) {
      capitalizedCount++;
    }
  }
  // Calculate density: capitalized words per 1000 tokens
  const density = (capitalizedCount / tokens) * 1000;

  // Text density threshold is typically lower than JSON
  // A well-written article might have 5-10% named entities
  // Half the JSON threshold applied
  return density > threshold * 500;
}

/**
 * Split a JSON value into chunks while preserving structure.
 * Arrays split at element boundaries; objects split at top-level key boundaries.
 */
export function chunkJsonData(
  data: unknown,
  chunkSizeTokens?: number,
  overlapTokens?: number,
): string[] {
  const chunkSizeChars = tokensToChars(chunkSizeTokens ?? CHUNK_TOKEN_SIZE);
  const overlapChars = tokensToChars(overlapTokens ?? CHUNK_OVERLAP_TOKENS);

  if (Array.isArray(data)) {
    return chunkJsonArray(data, chunkSizeChars, overlapChars);
  }
  // jsonLikelyDense gates scalars out; only objects reach here.
  return chunkJsonObject(data as Record<string, unknown>, chunkSizeChars, overlapChars);
}

function chunkJsonArray(
  data: unknown[],
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const parts = data.map((el) => JSON.stringify(el));
  return chunkJsonParts(parts, '[', ']', chunkSizeChars, overlapChars);
}

function chunkJsonObject(
  data: Record<string, unknown>,
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const parts = Object.entries(data).map(
    ([k, v]) => JSON.stringify(k) + ':' + JSON.stringify(v),
  );
  return chunkJsonParts(parts, '{', '}', chunkSizeChars, overlapChars);
}

function chunkJsonParts(
  parts: string[],
  open: string,
  close: string,
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const empty = open + close;
  if (parts.length === 0) return [empty];

  const chunks = accumulateChunks({
    items: parts,
    chunkSizeChars,
    baseSize: 2,
    separatorSize: 1,
    format: (ps) => open + ps.join(',') + close,
    seedNextChunk: (flushed) => {
      const items = takeTrailingItems(flushed, overlapChars, 2, 1);
      return { items, size: joinedJsonSize(items) };
    },
  });

  return chunks.length > 0 ? chunks : [empty];
}

function joinedJsonSize(parts: string[]): number {
  if (parts.length === 0) return 2;
  let size = 2 + (parts.length - 1);
  for (const p of parts) size += p.length;
  return size;
}

/**
 * Split text content at natural boundaries (paragraphs, sentences).
 * Includes overlap to capture entities at chunk boundaries.
 */
export function chunkTextContent(
  content: string,
  chunkSizeTokens?: number,
  overlapTokens?: number,
): string[] {
  const chunkSizeChars = tokensToChars(chunkSizeTokens ?? CHUNK_TOKEN_SIZE);
  const overlapChars = tokensToChars(overlapTokens ?? CHUNK_OVERLAP_TOKENS);

  if (content.length <= chunkSizeChars) return [content];

  const paragraphs = content
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const chunks = accumulateChunks({
    items: paragraphs,
    chunkSizeChars,
    baseSize: 0,
    separatorSize: 2,
    format: (ps) => ps.join('\n\n'),
    seedNextChunk: (flushed) => {
      const overlapText = getOverlapText(flushed.join('\n\n'), overlapChars);
      return overlapText
        ? { items: [overlapText], size: overlapText.length }
        : { items: [], size: 0 };
    },
    handleOversized: (p) => chunkBySentences(p, chunkSizeChars, overlapChars),
  });

  return chunks.length > 0 ? chunks : [content];
}

function splitIntoSentences(text: string): string[] {
  const sentences: string[] = [];
  let start = 0;

  for (let i = 0; i < text.length; i++) {
    if (!isSentenceEnd(text, i)) continue;
    let j = i + 1;
    while (j < text.length && /\s/.test(text[j])) j++;
    const sentence = text.substring(start, j).trim();
    if (sentence) sentences.push(sentence);
    start = j;
    i = j - 1;
  }

  if (start < text.length) {
    const tail = text.substring(start).trim();
    if (tail) sentences.push(tail);
  }

  return sentences;
}

function chunkBySentences(
  text: string,
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const sentences = splitIntoSentences(text)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  return accumulateChunks({
    items: sentences,
    chunkSizeChars,
    baseSize: 0,
    separatorSize: 1,
    format: (sents) => sents.join(' '),
    seedNextChunk: (flushed) => {
      const overlapText = getOverlapText(flushed.join(' '), overlapChars);
      return overlapText
        ? { items: [overlapText], size: overlapText.length }
        : { items: [], size: 0 };
    },
    handleOversized: (s) => chunkBySize(s, chunkSizeChars, overlapChars),
  });
}

function chunkBySize(
  text: string,
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + chunkSizeChars, text.length);

    if (end < text.length) {
      const slice = text.substring(start, end);
      const relSpace = slice.lastIndexOf(' ');
      if (relSpace > 0) {
        end = start + relSpace;
      }
    }

    chunks.push(text.substring(start, end).trim());

    const minProgress = Math.max(1, chunkSizeChars - overlapChars);
    start = Math.max(start + minProgress, end - overlapChars);
  }

  return chunks;
}

function getOverlapText(text: string, overlapChars: number): string {
  if (text.length <= overlapChars) return text;

  const overlapStart = text.length - overlapChars;
  const spaceIdx = text.indexOf(' ', overlapStart);
  if (spaceIdx !== -1) return text.substring(spaceIdx + 1);
  return text.substring(overlapStart);
}

/**
 * Split conversation content on speaker-turn boundaries.
 *
 * Callers must pre-format content as "Speaker: message" lines, which
 * addMessageEpisodes normalization guarantees.
 *
 * TODO(examine): pass MessageTurn[] through to a chunkMessageTurns variant to drop
 * the serialize -> re-split roundtrip. Requires widening the normalized
 * episode content type to carry structured turns alongside the string form.
 */
export function chunkMessageContent(
  content: string,
  chunkSizeTokens?: number,
  overlapTokens?: number,
): string[] {
  const chunkSizeChars = tokensToChars(chunkSizeTokens ?? CHUNK_TOKEN_SIZE);
  const overlapChars = tokensToChars(overlapTokens ?? CHUNK_OVERLAP_TOKENS);

  if (content.length <= chunkSizeChars) return [content];

  return chunkSpeakerMessages(content, chunkSizeChars, overlapChars);
}

function chunkSpeakerMessages(
  content: string,
  chunkSizeChars: number,
  overlapChars: number,
): string[] {
  const messages = content
    .split(/(?=^[A-Za-z_][A-Za-z0-9_\s]*:)/m)
    .map((m) => m.trim())
    .filter((m) => m.length > 0);

  if (messages.length === 0) return [content];

  const chunks = accumulateChunks({
    items: messages,
    chunkSizeChars,
    baseSize: 0,
    separatorSize: 1,
    format: (ms) => ms.join('\n'),
    seedNextChunk: (flushed) => {
      const items = takeTrailingItems(flushed, overlapChars, 0, 1);
      const size =
        items.reduce((sum, m) => sum + m.length, 0) + Math.max(0, items.length - 1);
      return { items, size };
    },
    handleOversized: (m) => [m],
  });

  return chunks.length > 0 ? chunks : [content];
}

type ChunkConfig = {
  items: string[];
  chunkSizeChars: number;
  baseSize: number;
  separatorSize: number;
  format: (items: string[]) => string;
  seedNextChunk: (flushed: string[]) => { items: string[]; size: number };
  handleOversized?: (item: string) => string[];
};

function accumulateChunks(cfg: ChunkConfig): string[] {
  const chunks: string[] = [];
  let current: string[] = [];
  let currentSize = cfg.baseSize;

  for (const item of cfg.items) {
    if (cfg.handleOversized && item.length > cfg.chunkSizeChars) {
      if (current.length > 0) {
        chunks.push(cfg.format(current));
        current = [];
        currentSize = cfg.baseSize;
      }
      chunks.push(...cfg.handleOversized(item));
      continue;
    }

    const sepBefore = current.length === 0 ? 0 : cfg.separatorSize;
    if (
      current.length > 0 &&
      currentSize + sepBefore + item.length > cfg.chunkSizeChars
    ) {
      chunks.push(cfg.format(current));
      const seed = cfg.seedNextChunk(current);
      current = seed.items;
      currentSize = seed.size;
    }

    const sep = current.length === 0 ? 0 : cfg.separatorSize;
    current.push(item);
    currentSize += sep + item.length;
  }

  if (current.length > 0) {
    chunks.push(cfg.format(current));
  }
  return chunks;
}

function takeTrailingItems(
  items: string[],
  budget: number,
  baseSize = 0,
  separatorSize = 1,
): string[] {
  if (items.length === 0) return [];
  const result: string[] = [];
  let size = baseSize;

  for (let i = items.length - 1; i >= 0; i--) {
    const itemSize = items[i].length;
    const sep = result.length === 0 ? 0 : separatorSize;
    if (size + sep + itemSize > budget) break;
    result.unshift(items[i]);
    size += sep + itemSize;
  }
  return result;
}
