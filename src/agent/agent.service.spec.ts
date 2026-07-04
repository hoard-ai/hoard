import { SubQueryType } from '@/knowledge-graph/search/types';

import type { AgentStep } from './agent.service';
import { buildAgentStepValidator } from './agent.service';

describe('buildAgentStepValidator', () => {
  const shown = new Set([0, 1, 2]);

  it('passes an answer whose grounding indices were all shown', () => {
    const validate = buildAgentStepValidator(shown, false);
    const step: AgentStep = {
      action: 'answer',
      text: 'frogs are dumb',
      groundingIndices: [0, 2],
    };
    expect(validate(step)).toEqual([]);
  });

  it('flags an answer grounded on an index that was not shown', () => {
    const validate = buildAgentStepValidator(shown, false);
    const step: AgentStep = { action: 'answer', text: 'x', groundingIndices: [0, 9] };
    const violations = validate(step);
    expect(violations.length).toBeGreaterThan(0);
    expect(violations.some((v) => v.code === 'agent.unknown-grounding-index')).toBe(true);
  });

  it('allows an answer with empty grounding (no-information case)', () => {
    const validate = buildAgentStepValidator(shown, false);
    const step: AgentStep = {
      action: 'answer',
      text: 'I do not have that',
      groundingIndices: [],
    };
    expect(validate(step)).toEqual([]);
  });

  it('flags an answer missing text', () => {
    const validate = buildAgentStepValidator(shown, false);
    const step: AgentStep = { action: 'answer', groundingIndices: [0] };
    expect(validate(step).some((v) => v.code === 'agent.answer-missing-text')).toBe(true);
  });

  it('passes a search with one or more areas (structure is Zod-enforced)', () => {
    const validate = buildAgentStepValidator(shown, false);
    const step: AgentStep = {
      action: 'search',
      searches: [
        {
          originalQuery: 'what do I think of frogs',
          queries: [{ type: SubQueryType.lex, text: 'frogs' }],
          limit: 8,
        },
      ],
    };
    expect(validate(step)).toEqual([]);
  });

  it('forces an answer once searches are exhausted', () => {
    const validate = buildAgentStepValidator(shown, true);
    const step: AgentStep = {
      action: 'search',
      searches: [
        {
          originalQuery: 'frogs',
          queries: [{ type: SubQueryType.lex, text: 'frogs' }],
          limit: 5,
        },
      ],
    };
    expect(validate(step).some((v) => v.code === 'agent.must-answer')).toBe(true);
  });
});
