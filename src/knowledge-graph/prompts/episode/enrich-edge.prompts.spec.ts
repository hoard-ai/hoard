import { HumanMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { KG_REFERENCE_TIME, KG_TEST_GRAPH_ID, KgNodeFactory } from '@/test/factories';

import {
  buildEnrichEdgeMessages,
  buildEnrichEdgeSchema,
  buildEnrichEdgeValidator,
  buildTimeReferenceTable,
} from './enrich-edge.prompts';

describe('buildTimeReferenceTable', () => {
  // 2024-03-31 is a Sunday in a leap year, month-end - exercises every edge case.
  const table = buildTimeReferenceTable(new Date('2024-03-31T14:30:00Z'));

  it('states the weekday, month length, and leap-year February', () => {
    expect(table).toContain('today is a Sunday');
    expect(table).toContain('this month (March) has 31 days');
    expect(table).toContain('February this year has 29 days (leap year)');
  });

  it('clamps month offsets to the end of the target month', () => {
    expect(table).toContain('1 month ago: 2024-02-29T00:00:00Z'); // leap + 31->29
    expect(table).toContain('6 months ago: 2023-09-30T00:00:00Z'); // 31->30
  });

  it('anchors week/month/quarter boundaries with a Monday week start', () => {
    expect(table).toContain('start of this week (Monday): 2024-03-25T00:00:00Z');
    expect(table).toContain('start of this month: 2024-03-01T00:00:00Z');
    expect(table).toContain(
      'current quarter: Q1 (2024-01-01T00:00:00Z to 2024-03-31T00:00:00Z)',
    );
  });

  it('resolves day offsets at midnight UTC', () => {
    expect(table).toContain('yesterday: 2024-03-30T00:00:00Z');
    expect(table).toContain('tomorrow: 2024-04-01T00:00:00Z');
  });

  it('marks a non-leap year and clamps to a 30-day month', () => {
    const t = buildTimeReferenceTable(new Date('2023-05-31T09:00:00Z'));
    expect(t).toContain('February this year has 28 days');
    expect(t).not.toContain('leap year');
    expect(t).toContain('1 month ago: 2023-04-30T00:00:00Z'); // May 31 -> Apr 30
  });
});

describe('buildEnrichEdgeSchema', () => {
  it('is temporal-only when no custom schema is given', () => {
    const schema = buildEnrichEdgeSchema();
    const parsed = schema.parse({ validAt: '2025-01-01T00:00:00Z', invalidAt: null });
    expect(parsed).toEqual({ validAt: '2025-01-01T00:00:00Z', invalidAt: null });
    expect('attributes' in (parsed as object)).toBe(false);
  });

  it('nests custom attributes under an attributes key', () => {
    const schema = buildEnrichEdgeSchema(z.object({ role: z.string() }));
    const parsed = schema.parse({
      validAt: null,
      invalidAt: null,
      attributes: { role: 'CEO' },
    });
    expect(parsed).toEqual({
      validAt: null,
      invalidAt: null,
      attributes: { role: 'CEO' },
    });
  });

  it('keeps a custom validAt field isolated under attributes (collision-proof)', () => {
    // A custom fact type that literally defines validAt must not clobber the
    // edge's temporal validAt - it lands under attributes.
    const schema = buildEnrichEdgeSchema(z.object({ validAt: z.string() }));
    const parsed = schema.parse({
      validAt: '2025-01-01T00:00:00Z',
      invalidAt: null,
      attributes: { validAt: 'CUSTOM' },
    }) as { validAt: string | null; attributes: { validAt: string } };
    expect(parsed.validAt).toBe('2025-01-01T00:00:00Z');
    expect(parsed.attributes.validAt).toBe('CUSTOM');
  });
});

describe('buildEnrichEdgeValidator', () => {
  const validate = buildEnrichEdgeValidator();

  it('passes when a bound is missing', () => {
    expect(validate({ validAt: '2025-01-01T00:00:00Z', invalidAt: null })).toEqual([]);
  });

  it('flags an inverted validity interval', () => {
    expect(
      validate({ validAt: '2025-06-01T00:00:00Z', invalidAt: '2025-01-01T00:00:00Z' })
        .length,
    ).toBeGreaterThan(0);
  });
});

describe('buildEnrichEdgeMessages', () => {
  const episode = KgNodeFactory.createEpisodicNode({
    name: 'Ep',
    content: 'Alice works at Acme.',
    graphId: KG_TEST_GRAPH_ID,
  });

  const human = (msgs: ReturnType<typeof buildEnrichEdgeMessages>): string =>
    msgs.find((m) => m instanceof HumanMessage)!.content as string;

  it('includes FACT, CURRENT EPISODE and REFERENCE TIME and omits EXISTING ATTRIBUTES when untyped', () => {
    const content = human(
      buildEnrichEdgeMessages({
        fact: 'Alice works at Acme',
        episode,
        referenceTime: KG_REFERENCE_TIME,
        hasCustomAttributes: false,
      }),
    );
    expect(content).toContain('<FACT>');
    expect(content).toContain('<CURRENT EPISODE>');
    expect(content).toContain('<REFERENCE TIME>');
    expect(content).toContain('<TIME REFERENCE>');
    expect(content).not.toContain('<EXISTING ATTRIBUTES>');
  });

  it('includes EXISTING ATTRIBUTES when typed', () => {
    const content = human(
      buildEnrichEdgeMessages({
        fact: 'Alice works at Acme',
        episode,
        referenceTime: KG_REFERENCE_TIME,
        existingAttributes: { role: 'CEO' },
        hasCustomAttributes: true,
      }),
    );
    expect(content).toContain('<EXISTING ATTRIBUTES>');
  });
});
