// pgvector wire format is the text representation '[0.1,0.2,...]'. Prisma's
// $queryRaw / $executeRaw must serialize embeddings to this format on write
// and parse them back from `name_embedding::text` projections on read.

export const toPgVector = (v: readonly number[] | null): string | null =>
  v === null ? null : `[${v.join(',')}]`;

export const fromPgVector = (s: string | null | undefined): number[] | null => {
  if (s === null || s === undefined) return null;
  return JSON.parse(s) as number[];
};

const MAX_BIND_PARAMS = 65535;

/**
 * Splits `items` into chunks whose multi-row statement stays within Postgres'
 * bind-parameter ceiling. `paramsPerRow` is the number of bind parameters one
 * row contributes to the statement.
 */
export function chunkForBindParams<T>(items: T[], paramsPerRow: number): T[][] {
  const maxRows = Math.max(1, Math.floor(MAX_BIND_PARAMS / paramsPerRow));
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += maxRows) {
    chunks.push(items.slice(i, i + maxRows));
  }
  return chunks;
}

/**
 * Last-wins dedup by id. A multi-row INSERT ... ON CONFLICT DO UPDATE errors
 * ("cannot affect row a second time") if the same conflict target appears twice
 * in one statement, so callers dedup before building the VALUES list.
 */
export function dedupeById<T extends { id: string }>(items: T[]): T[] {
  return [...new Map(items.map((item) => [item.id, item])).values()];
}
