import { RelationshipTypeSchema } from '@/knowledge-graph/types';

import {
  buildExtractEdgesValidator,
  ExtractedEdgesWithCorefSchema,
} from './extract-edges.prompts';

const rel = (s: string) => RelationshipTypeSchema.parse(s);

describe('buildExtractEdgesValidator', () => {
  const validate = buildExtractEdgesValidator({
    nodes: [{ name: 'Alice' }, { name: 'Bob' }],
  });

  const edge = (overrides: Partial<{ source: number; target: number }>) => ({
    sourceEntityIdx: overrides.source ?? 0,
    targetEntityIdx: overrides.target ?? 1,
    relationType: rel('WORKS_WITH'),
    fact: 'Alice works with Bob',
  });

  it('passes valid endpoints', () => {
    expect(validate({ edges: [edge({})] })).toEqual([]);
  });

  it('flags source idx out of range', () => {
    expect(validate({ edges: [edge({ source: 5 })] }).length).toBeGreaterThan(0);
  });

  it('flags target idx out of range', () => {
    expect(validate({ edges: [edge({ target: 5 })] }).length).toBeGreaterThan(0);
  });

  it('passes a self-loop (single-entity fact channel)', () => {
    expect(validate({ edges: [edge({ target: 0 })] })).toEqual([]);
  });

  it('passes an in-range usedCoreferences entityIdx', () => {
    expect(
      validate({
        edges: [edge({})],
        usedCoreferences: [
          {
            surfaceForm: 'she',
            entityIdx: 1,
            locatingQuote: 'she left',
            unresolvedReferenceIdx: null,
          },
        ],
      }),
    ).toEqual([]);
  });

  it('flags an out-of-range usedCoreferences entityIdx', () => {
    expect(
      validate({
        edges: [edge({})],
        usedCoreferences: [
          {
            surfaceForm: 'she',
            entityIdx: 5,
            locatingQuote: 'she left',
            unresolvedReferenceIdx: null,
          },
        ],
      }).length,
    ).toBeGreaterThan(0);
  });

  it('requires a locatingQuote on each usedCoreferences entry', () => {
    const result = ExtractedEdgesWithCorefSchema.safeParse({
      edges: [],
      usedCoreferences: [{ surfaceForm: 'she', entityIdx: 0 }],
    });
    expect(result.success).toBe(false);
  });

  describe('unresolvedReferenceIdx claims', () => {
    const validateWithRefs = buildExtractEdgesValidator({
      nodes: [{ name: 'Alice' }, { name: 'Bob' }],
      unresolvedReferences: [{ surfaceForm: 'she', locatingQuote: 'she left early' }],
    });

    it('passes an in-range claim', () => {
      expect(
        validateWithRefs({
          edges: [edge({})],
          usedCoreferences: [
            {
              surfaceForm: 'She',
              entityIdx: 0,
              locatingQuote: 'She left',
              unresolvedReferenceIdx: 0,
            },
          ],
        }),
      ).toEqual([]);
    });

    it('flags an out-of-range unresolvedReferenceIdx', () => {
      expect(
        validateWithRefs({
          edges: [edge({})],
          usedCoreferences: [
            {
              surfaceForm: 'she',
              entityIdx: 0,
              locatingQuote: 'she left',
              unresolvedReferenceIdx: 3,
            },
          ],
        }).length,
      ).toBeGreaterThan(0);
    });

    it('passes a null claim regardless of the reference list', () => {
      expect(
        validateWithRefs({
          edges: [edge({})],
          usedCoreferences: [
            {
              surfaceForm: 'the doctor',
              entityIdx: 1,
              locatingQuote: 'the doctor arrived',
              unresolvedReferenceIdx: null,
            },
          ],
        }),
      ).toEqual([]);
    });
  });
});
