import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { Inject, Injectable } from '@nestjs/common';
import { z } from 'zod';

import { Uuid } from '@/common/schemas';
import { SearchService } from '@/knowledge-graph/search';
import {
  AgenticSearchResults,
  SearchGroupSchema,
  SearchResults,
} from '@/knowledge-graph/search/types';
import { invokeStructured, Violation } from '@/llm';
import { LlmService } from '@/llm/llm.service';
import { LLM_TRACER, type LlmContext, type LlmTracer } from '@/observability';

const MAX_SEARCH_ITERATIONS = 5;
const MAX_SEARCH_GROUPS = 4;

const AgentStepSchema = z.object({
  action: z
    .enum(['search', 'answer'])
    .describe('Whether to search for more information or answer now.'),
  searches: z
    .array(SearchGroupSchema)
    .min(1)
    .max(MAX_SEARCH_GROUPS)
    .optional()
    .describe(
      `For action "search": 1-${MAX_SEARCH_GROUPS} independent areas to search, each with its own query, sub-queries, and result limit.`,
    ),
  text: z.string().optional().describe('For action "answer": the answer to the user.'),
  groundingIndices: z
    .array(z.int().nonnegative())
    .optional()
    .describe(
      'For action "answer": the [index] numbers of the candidates you used. Only indices shown to you.',
    ),
});

export type AgentStep = z.infer<typeof AgentStepSchema>;

type Candidate = {
  kind: 'fact' | 'entity' | 'episode';
  id: Uuid;
  text: string;
  sources?: Uuid[];
  // The search area (originalQuery) this came from; absent for prefetch results.
  group?: string;
};

export type AgentReply = { text: string; grounding: Uuid[] };

const AgentState = Annotation.Root({
  query: Annotation<string>({ reducer: (_prev, next) => next, default: () => '' }),
  candidates: Annotation<Candidate[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
  step: Annotation<AgentStep | null>({
    reducer: (_prev, next) => next,
    default: () => null,
  }),
  answer: Annotation<string>({ reducer: (_prev, next) => next, default: () => '' }),
  grounding: Annotation<Uuid[]>({ reducer: (_prev, next) => next, default: () => [] }),
  iterations: Annotation<number>({
    reducer: (prev, next) => prev + next,
    default: () => 0,
  }),
});

type AgentStateType = typeof AgentState.State;

@Injectable()
export class AgentService {
  constructor(
    private readonly searchService: SearchService,
    private readonly llmService: LlmService,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  getAgentInfo(userId: Uuid) {
    return { userId };
  }

  async chat(params: {
    userId: Uuid;
    graphIds: Uuid[];
    message: string;
  }): Promise<AgentReply> {
    const { userId, graphIds, message } = params;

    // Bypass the concurrency limiter so a live conversation
    // never queues behind background ingestion / bulk work.
    const model = await this.llmService.getActiveModel(userId, { rateLimited: false });
    const ctx: LlmContext = {
      userId,
      tags: ['agent', ...graphIds.map((id) => `group:${id}`)],
      metadata: { query: message.slice(0, 200) },
    };
    const callbacks = this.llmTracer.getCallbacks(ctx);

    const prefetchNode = async (
      state: AgentStateType,
    ): Promise<Partial<AgentStateType>> => {
      const results = await this.searchService.prefetch({ query: state.query, graphIds });
      return { candidates: prefetchToCandidates(results) };
    };

    const decideNode = async (
      state: AgentStateType,
    ): Promise<Partial<AgentStateType>> => {
      const shownIndices = new Set(state.candidates.map((_c, i) => i));
      const mustAnswer = state.iterations >= MAX_SEARCH_ITERATIONS;

      const step = await invokeStructured(
        model,
        AgentStepSchema,
        [
          new SystemMessage(buildSystemPrompt(mustAnswer)),
          new HumanMessage(buildDecideHuman(state.query, state.candidates)),
        ],
        {
          runName: 'agent.decide',
          tags: ['agent', 'decide'],
          callbacks,
          validate: buildAgentStepValidator(shownIndices, mustAnswer),
        },
      );
      return { step };
    };

    const searchNode = async (
      state: AgentStateType,
    ): Promise<Partial<AgentStateType>> => {
      const groups = state.step?.searches ?? [];
      const resultSets = await Promise.all(
        groups.map((g) =>
          this.searchService.searchExpanded({
            originalQuery: g.originalQuery,
            queries: g.queries,
            limit: g.limit,
            graphIds,
          }),
        ),
      );
      // First-group-wins dedup, seeded from candidates already shown (prefetch
      // + earlier iterations), so an id keeps its original index and label.
      const seen = new Set(state.candidates.map((c) => c.id));
      const fresh: Candidate[] = [];
      for (let gi = 0; gi < groups.length; gi++) {
        for (const c of agenticToCandidates(resultSets[gi])) {
          if (seen.has(c.id)) continue;
          seen.add(c.id);
          fresh.push({ ...c, group: groups[gi].originalQuery });
        }
      }
      return { candidates: fresh, iterations: 1 };
    };

    const finalizeNode = (state: AgentStateType): Partial<AgentStateType> => {
      const step = state.step;
      const grounding = (step?.groundingIndices ?? [])
        .map((i) => state.candidates[i]?.id)
        .filter((id): id is Uuid => id !== undefined);
      return { answer: step?.text ?? '', grounding };
    };

    const graph = new StateGraph(AgentState)
      .addNode('prefetch', prefetchNode)
      .addNode('decide', decideNode)
      .addNode('search', searchNode)
      .addNode('finalize', finalizeNode)
      .addEdge(START, 'prefetch')
      .addEdge('prefetch', 'decide')
      .addConditionalEdges('decide', routeDecision, ['search', 'finalize'])
      .addEdge('search', 'decide')
      .addEdge('finalize', END)
      .compile();

    const result = await graph.invoke({ query: message });
    return { text: result.answer, grounding: result.grounding };
  }
}

function routeDecision(state: AgentStateType): 'search' | 'finalize' {
  return state.step?.action === 'search' ? 'search' : 'finalize';
}

export function buildAgentStepValidator(
  shownIndices: Set<number>,
  mustAnswer: boolean,
): (step: AgentStep) => Violation[] {
  return (step) => {
    const violations: Violation[] = [];

    if (mustAnswer && step.action !== 'answer') {
      violations.push({
        code: 'agent.must-answer',
        message: 'no searches remain; action must be "answer"',
      });
      return violations;
    }

    if (step.action === 'answer') {
      if (!step.text) {
        violations.push({
          code: 'agent.answer-missing-text',
          message: 'action "answer" requires text',
        });
      }
      for (const index of step.groundingIndices ?? []) {
        if (!shownIndices.has(index)) {
          violations.push({
            code: 'agent.unknown-grounding-index',
            message: `grounding index ${index} is not in the candidate set`,
          });
        }
      }
    }
    return violations;
  };
}

const SYSTEM_PROMPT = `You are Hoard, an assistant with access to the user's personal knowledge graph.
When the user sends a message, an automatic pre-fetch retrieves candidate items from the graph for it and shows them to you numbered. 
This runs once, at the start; you did not request it. Any further candidates come from searches you choose to run.

Candidates are grouped under headers showing which search they answer; the [index] numbers run continuously across all groups.

Decide each step:
- If the candidates are sufficient to answer, set action to "answer" and write the answer in "text". 
List the [index] numbers you actually used in "groundingIndices", using only indices that appear in the candidate list.
- If they are insufficient, set action to "search" and provide "searches": 
one entry per distinct area you need to cover (search several areas at once rather than one at a time). Each entry has:
  - originalQuery: the question this area answers, in natural language.
  - queries: 2-4 typed sub-queries:
    - lex: keyword full-text search; supports OR, "quoted phrases", and -negation (e.g. frogs OR amphibians "pretty dumb" -dogs).
    - vec: a semantic rephrasing of the question.
    - hyde: a hypothetical answer to the question (its wording is embedded for semantic search).
  - limit: how many results you want back for this area. Choose based on how much you need; ask for more for broad areas, fewer for narrow lookups.

Ground every claim in the candidates you cite. If the message needs no knowledge from 
the graph (a greeting, small talk, or other trivial message), just answer and omit "groundingIndices". 
If the graph simply does not contain the answer, say so in "text" with an empty "groundingIndices".`;

function buildSystemPrompt(mustAnswer: boolean): string {
  if (!mustAnswer) return SYSTEM_PROMPT;
  return `${SYSTEM_PROMPT}\n\nYou have no searches remaining. You MUST set action to "answer" now, using whatever candidates are available.`;
}

function buildDecideHuman(query: string, candidates: Candidate[]): string {
  return `User question: ${query}\n\nCandidates:\n${formatCandidates(candidates)}`;
}

function formatCandidates(candidates: Candidate[]): string {
  if (candidates.length === 0) return 'No candidates retrieved yet.';

  // Bucket by group label while keeping each candidate's global index, so the
  // grounding-index invariant (index = position in the flat candidate array)
  // holds across the sectioned display.
  const sections = new Map<string, string[]>();
  candidates.forEach((c, i) => {
    const header = c.group
      ? `## Search: "${c.group}"`
      : '## Automatic results (from your message)';
    const sources = c.sources?.length ? ` (sources: ${c.sources.join(', ')})` : '';
    const line = `[${i}] (${c.kind}) ${c.text}${sources}`;
    const lines = sections.get(header);
    if (lines) lines.push(line);
    else sections.set(header, [line]);
  });

  return [...sections.entries()]
    .map(([header, lines]) => `${header}\n${lines.join('\n')}`)
    .join('\n\n');
}

function prefetchToCandidates(results: SearchResults): Candidate[] {
  const candidates: Candidate[] = [];
  for (const e of results.edges) {
    candidates.push({ kind: 'fact', id: e.id, text: e.fact, sources: e.episodes });
  }
  for (const n of results.nodes) {
    candidates.push({ kind: 'entity', id: n.id, text: `${n.name}: ${n.summary}` });
  }
  return candidates;
}

function agenticToCandidates(results: AgenticSearchResults): Candidate[] {
  const candidates: Candidate[] = [];
  for (const e of results.edges) {
    candidates.push({ kind: 'fact', id: e.id, text: e.fact, sources: e.episodes });
  }
  for (const n of results.nodes) {
    candidates.push({ kind: 'entity', id: n.id, text: `${n.name}: ${n.summary}` });
  }
  for (const ep of results.episodes) {
    const snippet = results.episodeSnippets.get(ep.id) ?? ep.content.slice(0, 300);
    candidates.push({ kind: 'episode', id: ep.id, text: snippet });
  }
  return candidates;
}
