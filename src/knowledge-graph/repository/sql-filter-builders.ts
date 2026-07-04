import { Prisma } from '@generated/prisma/client';

import type { SearchFilters } from '../search/types';
import { TemporalComparison } from '../search/types';

// Postgres equivalents of cypher-filter-builders. Each returns a Prisma.Sql
// fragment (or empty) ready to be inlined inside a $queryRaw template. The
// caller composes the fragment with ` AND ${fragment}` after its base WHERE.

const empty: Prisma.Sql = Prisma.empty;

function buildTemporalConditions(
  temporalFilters: SearchFilters['temporalFilters'],
  alias: string,
): Prisma.Sql {
  if (!temporalFilters || temporalFilters.length === 0) return empty;

  const groupSql: Prisma.Sql[] = [];

  for (const group of temporalFilters) {
    const andParts: Prisma.Sql[] = [];

    for (const tf of group) {
      // alias.field - both are safe identifiers (alias is a hardcoded string,
      // field is enum-validated). Embed as raw SQL, not as a parameter.
      const col = Prisma.raw(`"${alias}"."${tf.field}"`);

      if (tf.op === TemporalComparison.isNull) {
        andParts.push(Prisma.sql`${col} IS NULL`);
      } else if (tf.op === TemporalComparison.isNotNull) {
        andParts.push(Prisma.sql`${col} IS NOT NULL`);
      } else {
        // Op enum values map 1:1 to SQL operators (= < > <= >= <>)
        const op = Prisma.raw(tf.op);
        andParts.push(Prisma.sql`${col} ${op} ${tf.value}`);
      }
    }

    if (andParts.length === 1) {
      groupSql.push(andParts[0]);
    } else if (andParts.length > 1) {
      groupSql.push(Prisma.sql`(${Prisma.join(andParts, ' AND ')})`);
    }
  }

  if (groupSql.length === 0) return empty;
  if (groupSql.length === 1) return groupSql[0];
  return Prisma.sql`(${Prisma.join(groupSql, ' OR ')})`;
}

// Builds `AND <conditions>` (with leading ` AND `) or empty Sql if no filters apply.
// Caller writes: `WHERE base_predicate ${buildNodeFilterClause(filters, 'n')}`.
export function buildNodeFilterClause(
  filters: SearchFilters | undefined,
  alias: string,
): Prisma.Sql {
  if (!filters) return empty;
  const parts: Prisma.Sql[] = [];

  if (filters.nodeLabels && filters.nodeLabels.length > 0) {
    const col = Prisma.raw(`"${alias}"."labels"`);
    parts.push(Prisma.sql`${col} && ${filters.nodeLabels}::text[]`);
  }
  const temporal = buildTemporalConditions(filters.temporalFilters, alias);
  if (temporal !== empty) parts.push(temporal);

  if (parts.length === 0) return empty;
  return Prisma.sql` AND ${Prisma.join(parts, ' AND ')}`;
}

export function buildEdgeFilterClause(
  filters: SearchFilters | undefined,
  alias: string,
): Prisma.Sql {
  if (!filters) return empty;
  const parts: Prisma.Sql[] = [];

  if (filters.edgeIds && filters.edgeIds.length > 0) {
    const col = Prisma.raw(`"${alias}"."id"`);
    parts.push(Prisma.sql`${col} = ANY(${filters.edgeIds}::uuid[])`);
  }
  if (filters.edgeTypes && filters.edgeTypes.length > 0) {
    const col = Prisma.raw(`"${alias}"."name"`);
    parts.push(Prisma.sql`${col} = ANY(${filters.edgeTypes}::text[])`);
  }
  const temporal = buildTemporalConditions(filters.temporalFilters, alias);
  if (temporal !== empty) parts.push(temporal);

  if (parts.length === 0) return empty;
  return Prisma.sql` AND ${Prisma.join(parts, ' AND ')}`;
}

// TODO: examine if safe
/**
 * Builds a `websearch_to_tsquery` fragment for full-text search.
 *
 * websearch_to_tsquery never raises syntax errors (safe on raw agent input) and
 * supports unquoted AND, "quoted" phrases, the literal word OR, and `-negation`.
 * It does NOT support prefix-match (`*`) or weight labels - plain terms are
 * stemmed-exact.
 */
export function websearchTsquery(query: string): Prisma.Sql {
  return Prisma.sql`websearch_to_tsquery('english', ${query})`;
}
