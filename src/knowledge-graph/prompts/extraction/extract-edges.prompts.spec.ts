import { RelationshipTypeSchema } from '@/knowledge-graph/types';

import { buildExtractEdgesValidator } from './extract-edges.prompts';

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

  it('flags self-loop', () => {
    expect(validate({ edges: [edge({ target: 0 })] }).length).toBeGreaterThan(0);
  });
});
