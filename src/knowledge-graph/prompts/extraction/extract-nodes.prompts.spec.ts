import { z } from 'zod';

import { NodeNameSchema } from '@/knowledge-graph/types';

import {
  buildExtractNodesValidator,
  ExtractedEntitiesSchema,
} from './extract-nodes.prompts';

const n = (s: string) => NodeNameSchema.parse(s);

const types = {
  Person: { description: 'a person', schema: z.object({ x: z.string() }) },
  Place: { description: 'a place', schema: z.object({ x: z.string() }) },
};

describe('buildExtractNodesValidator', () => {
  it('passes when no entityTypes provided', () => {
    const validate = buildExtractNodesValidator({});
    expect(
      validate({
        extractedEntities: [
          {
            name: n('X'),
            entityTypeId: 999,
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
        unresolvedReferences: [],
      }),
    ).toEqual([]);
  });

  it('passes valid entityTypeId', () => {
    const validate = buildExtractNodesValidator({ entityTypes: types });
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Alice'),
            entityTypeId: 0,
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
          {
            name: n('Denver'),
            entityTypeId: 1,
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
        unresolvedReferences: [],
      }),
    ).toEqual([]);
  });

  it('passes when entityTypeId is omitted', () => {
    const validate = buildExtractNodesValidator({ entityTypes: types });
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Mystery'),
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
        unresolvedReferences: [],
      }),
    ).toEqual([]);
  });

  it('flags out-of-range entityTypeId', () => {
    const validate = buildExtractNodesValidator({ entityTypes: types });
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Alice'),
            entityTypeId: 999,
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
        unresolvedReferences: [],
      }).length,
    ).toBeGreaterThan(0);
  });

  it('flags negative entityTypeId', () => {
    const validate = buildExtractNodesValidator({ entityTypes: types });
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Alice'),
            entityTypeId: -1,
            identifyingDescription: '',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
        unresolvedReferences: [],
      }).length,
    ).toBeGreaterThan(0);
  });

  it('flags a bare-pronoun alias regardless of entityTypes (case-insensitive)', () => {
    const validate = buildExtractNodesValidator({});
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Priya Nair'),
            identifyingDescription: 'geneticist',
            aliases: ['Priya Nair', 'She'],
            referredToAsPronouns: ['she'],
          },
        ],
        unresolvedReferences: [],
      }).length,
    ).toBeGreaterThan(0);
  });

  it('passes definite-description and name-variant aliases', () => {
    const validate = buildExtractNodesValidator({});
    expect(
      validate({
        extractedEntities: [
          {
            name: n('Dr. Amara Osei'),
            identifyingDescription: 'lead researcher',
            aliases: ['Dr. Osei', 'the lead researcher', 'his larger vessel'],
            referredToAsPronouns: ['she'],
          },
        ],
        unresolvedReferences: [],
      }),
    ).toEqual([]);
  });
});

describe('coreference producer', () => {
  it('parses per-entity descriptors and top-level unresolved references', () => {
    const parsed = ExtractedEntitiesSchema.parse({
      extractedEntities: [
        {
          name: 'Dr. Amara Osei',
          identifyingDescription: 'lead researcher',
          aliases: ['Dr. Osei'],
          referredToAsPronouns: ['she'],
        },
      ],
      unresolvedReferences: [{ surfaceForm: 'they', locatingQuote: 'they later met' }],
    });
    expect(parsed.extractedEntities[0].aliases).toEqual(['Dr. Osei']);
    expect(parsed.extractedEntities[0].referredToAsPronouns).toEqual(['she']);
    expect(parsed.unresolvedReferences).toHaveLength(1);
  });

  it('rejects entities without descriptors', () => {
    const result = ExtractedEntitiesSchema.safeParse({
      extractedEntities: [{ name: 'Dr. Amara Osei' }],
      unresolvedReferences: [],
    });
    expect(result.success).toBe(false);
  });

  it('defaults referredToAsPronouns to empty when omitted', () => {
    const parsed = ExtractedEntitiesSchema.parse({
      extractedEntities: [
        {
          name: 'Belmont Arts Center',
          identifyingDescription: 'community arts venue',
          aliases: [],
        },
      ],
      unresolvedReferences: [],
    });
    expect(parsed.extractedEntities[0].referredToAsPronouns).toEqual([]);
  });
});
