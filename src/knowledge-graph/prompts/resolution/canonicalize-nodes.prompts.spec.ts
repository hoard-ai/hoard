import {
  buildCanonicalizeNodesMessages,
  buildCanonicalizeNodesValidator,
  CanonicalizeNodesSchema,
} from './canonicalize-nodes.prompts';

const entities = [
  {
    id: 0,
    name: 'Dr. Elena Marquez',
    labels: ['Entity'],
    identifyingDescription: 'a',
    aliases: [],
    referredToAsPronouns: ['she'],
  },
  {
    id: 1,
    name: 'biologist',
    labels: ['Entity'],
    identifyingDescription: 'b',
    aliases: ['the biologist'],
    referredToAsPronouns: [],
  },
  {
    id: 2,
    name: 'Valparaiso',
    labels: ['Entity'],
    identifyingDescription: 'c',
    aliases: [],
    referredToAsPronouns: [],
  },
];

describe('canonicalize-nodes prompts', () => {
  describe('CanonicalizeNodesSchema', () => {
    it('accepts groups of two or more ids and an empty group list', () => {
      expect(
        CanonicalizeNodesSchema.parse({ duplicateGroups: [[0, 1]] }).duplicateGroups,
      ).toEqual([[0, 1]]);
      expect(
        CanonicalizeNodesSchema.parse({ duplicateGroups: [] }).duplicateGroups,
      ).toEqual([]);
    });

    it('rejects singleton groups', () => {
      expect(CanonicalizeNodesSchema.safeParse({ duplicateGroups: [[0]] }).success).toBe(
        false,
      );
    });
  });

  describe('buildCanonicalizeNodesMessages', () => {
    it('renders each entity with its id, name, identifyingDescription, aliases, and pronouns', () => {
      const text = buildCanonicalizeNodesMessages({ entities })
        .map((m) => m.content as string)
        .join('\n');

      expect(text).toContain('id: 1');
      expect(text).toContain('biologist');
      expect(text).toContain('the biologist');
      expect(text).toContain('Dr. Elena Marquez');
      expect(text).toContain('referredToAsPronouns: ["she"]');
    });
  });

  describe('buildCanonicalizeNodesValidator', () => {
    const validate = buildCanonicalizeNodesValidator({ entities });

    it('passes disjoint in-range groups', () => {
      expect(validate({ duplicateGroups: [[0, 1]] })).toEqual([]);
      expect(validate({ duplicateGroups: [] })).toEqual([]);
    });

    it('flags an out-of-range id', () => {
      expect(validate({ duplicateGroups: [[0, 9]] }).length).toBeGreaterThan(0);
    });

    it('flags an id appearing in two groups', () => {
      expect(
        validate({
          duplicateGroups: [
            [0, 1],
            [1, 2],
          ],
        }).length,
      ).toBeGreaterThan(0);
    });

    it('flags an id repeated within one group', () => {
      expect(validate({ duplicateGroups: [[1, 1]] }).length).toBeGreaterThan(0);
    });
  });
});
