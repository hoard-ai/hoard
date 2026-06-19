import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Inject, Injectable } from '@nestjs/common';

import type { Uuid } from '@/common/schemas';
import { LlmService } from '@/llm/llm.service';
import {
  LLM_TRACER,
  type LlmContext,
  type LlmTracer,
  metricsOnResult,
  Span,
  type SpanMetrics,
} from '@/observability';

import { EmbeddingService } from '../embedding';
import { Community, EntityEdge, EntityNode, EpisodicNode } from '../models';
import {
  CommunityRepository,
  EntityEdgeRepository,
  EntityNodeRepository,
  EpisodicNodeRepository,
} from '../repository/repositories';
import {
  SearchByBfsParamsSchema,
  SearchBySimilarityParamsSchema,
  SearchByTextParamsSchema,
} from '../types';
import {
  crossEncoderReranker,
  episodeMentionsReranker,
  mmr,
  nodeDistanceReranker,
  rrf,
  weightedRrf,
} from './search-utils';
import {
  AGENTIC_CANDIDATE_LIMIT,
  AgenticSearchResults,
  CommunityReranker,
  type CommunitySearchConfig,
  CommunitySearchMethod,
  DEFAULT_MIN_SCORE,
  EdgeReranker,
  type EdgeSearchConfig,
  EdgeSearchMethod,
  emptyAgenticResults,
  emptySearchResults,
  EpisodeReranker,
  type EpisodeSearchConfig,
  EpisodeSearchMethod,
  ExpandedQuery,
  NodeReranker,
  type NodeSearchConfig,
  NodeSearchMethod,
  PREFETCH_BFS_DEPTH,
  PREFETCH_CANDIDATES,
  PREFETCH_LIMIT,
  RRF_ORIGINAL_WEIGHT,
  SearchConfigInput,
  SearchFilters,
  SearchOptions,
  type SearchOptionsInput,
  SearchOptionsSchema,
  SearchResults,
  SubQueryType,
} from './types';

@Injectable()
export class SearchService {
  constructor(
    private readonly llmService: LlmService,
    private readonly embeddingService: EmbeddingService,
    private readonly entityNodeRepository: EntityNodeRepository,
    private readonly entityEdgeRepository: EntityEdgeRepository,
    private readonly episodicNodeRepository: EpisodicNodeRepository,
    private readonly communityRepository: CommunityRepository,
    @Inject(LLM_TRACER) private readonly llmTracer: LlmTracer,
  ) {}

  async searchFromNodes(options: {
    nodeIds: Uuid[];
    query: string;
    graphIds: Uuid[];
    config: SearchConfigInput;
    userId: Uuid;
    filters?: SearchFilters;
  }): Promise<SearchResults> {
    return this.search({
      query: options.query,
      graphIds: options.graphIds,
      config: options.config,
      userId: options.userId,
      filters: options.filters,
      originNodeIds: options.nodeIds,
      centerNodeId: options.nodeIds[0],
    });
  }

  async search(options: SearchOptionsInput): Promise<SearchResults> {
    const { metrics: _m, ...rest } = await this.searchImpl(options);
    return rest;
  }

  // =============== TODO: QMD STYLE FUNCTIONS ====================================
  // Must evaluate these in order to derive some sort of superior hybrid
  // with graph-native Graphiti search. Need eval set first in order to do that. Delaying until then.
  // The agent should be able to traverse the graph.

  // ─── Pre-fetch (vector + one BFS hop) ─────────────────────────────────────────
  // Mode A: automatic, runs on every user message. Vector carries NL recall; one
  // BFS hop pulls in connected facts the vector pass missed. No FTS, no rerank.

  @Span('search.prefetch', { observationKind: 'retriever' })
  async prefetch(params: {
    query: string;
    graphIds: Uuid[];
    filters?: SearchFilters;
  }): Promise<SearchResults> {
    const { query, graphIds, filters } = params;
    if (!query.trim() || graphIds.length === 0) return emptySearchResults();

    const embedding = await this.embeddingService.embedText(query);
    if (!embedding) return emptySearchResults();

    const simParams = (limit: number) =>
      SearchBySimilarityParamsSchema.parse({
        embedding,
        graphIds,
        limit,
        minScore: DEFAULT_MIN_SCORE,
      });

    // Pass 1: vector over nodes + edges.
    const [vecNodes, vecEdges] = await Promise.all([
      this.entityNodeRepository.searchBySimilarity(
        simParams(PREFETCH_CANDIDATES),
        filters,
      ),
      this.entityEdgeRepository.searchBySimilarity(
        simParams(PREFETCH_CANDIDATES),
        filters,
      ),
    ]);

    const nodeMap = new Map<Uuid, EntityNode>(vecNodes.map((n) => [n.id, n]));
    const edgeMap = new Map<Uuid, EntityEdge>(vecEdges.map((e) => [e.id, e]));

    // Pass 2: one BFS hop seeded from the vector hits (nodes + edge endpoints).
    const seedNodeIds = [
      ...new Set<Uuid>([
        ...vecNodes.map((n) => n.id),
        ...vecEdges.flatMap((e) => [e.sourceNodeId, e.targetNodeId]),
      ]),
    ];

    let bfsNodeIds: Uuid[] = [];
    let bfsEdgeIds: Uuid[] = [];
    if (seedNodeIds.length > 0) {
      const bfsParams = SearchByBfsParamsSchema.parse({
        originNodeIds: seedNodeIds,
        graphIds,
        limit: PREFETCH_CANDIDATES,
        maxDepth: PREFETCH_BFS_DEPTH,
      });
      const [bfsNodes, bfsEdges] = await Promise.all([
        this.entityNodeRepository.searchByBfs(bfsParams, filters),
        this.entityEdgeRepository.searchByBfs(bfsParams, filters),
      ]);
      for (const n of bfsNodes) nodeMap.set(n.id, n);
      for (const e of bfsEdges) edgeMap.set(e.id, e);
      bfsNodeIds = bfsNodes.map((n) => n.id);
      bfsEdgeIds = bfsEdges.map((e) => e.id);
    }

    // Fuse vector + BFS per entity type (single query → no original weighting).
    const [nodeIds, nodeScores] = weightedRrf(
      [vecNodes.map((n) => n.id), bfsNodeIds].filter((l) => l.length > 0),
    );
    const [edgeIds, edgeScores] = weightedRrf(
      [vecEdges.map((e) => e.id), bfsEdgeIds].filter((l) => l.length > 0),
    );

    const nodeScoreById = new Map(nodeIds.map((id, i) => [id, nodeScores[i]]));
    const edgeScoreById = new Map(edgeIds.map((id, i) => [id, edgeScores[i]]));

    const topNodes = nodeIds
      .slice(0, PREFETCH_LIMIT)
      .map((id) => nodeMap.get(id))
      .filter((n): n is EntityNode => n !== undefined);
    const topEdges = edgeIds
      .slice(0, PREFETCH_LIMIT)
      .map((id) => edgeMap.get(id))
      .filter((e): e is EntityEdge => e !== undefined);

    return {
      edges: topEdges,
      edgeScores: new Map(topEdges.map((e) => [e.id, edgeScoreById.get(e.id) ?? 0])),
      nodes: topNodes,
      nodeScores: new Map(topNodes.map((n) => [n.id, nodeScoreById.get(n.id) ?? 0])),
      episodes: [],
      episodeScores: new Map(),
      communities: [],
      communityScores: new Map(),
    };
  }

  // ─── Agentic search (typed multi-query, no cross-encoder) ─────────────────────
  // Mode B: on-demand tool. The agent authors typed lex/vec/hyde sub-queries;
  // they fan out, fuse via weighted RRF, and return ranked candidates for the
  // agent to read and rerank. No cross-encoder.

  @Span('search.expanded', { observationKind: 'retriever' })
  async searchExpanded(params: {
    queries: ExpandedQuery[];
    originalQuery: string;
    limit: number;
    graphIds: Uuid[];
    filters?: SearchFilters;
  }): Promise<AgenticSearchResults> {
    const { queries, originalQuery, limit, graphIds, filters } = params;
    if (graphIds.length === 0) return emptyAgenticResults();

    const edgeMap = new Map<Uuid, EntityEdge>();
    const nodeMap = new Map<Uuid, EntityNode>();
    const episodeMap = new Map<Uuid, EpisodicNode>();
    const rankedLists: Uuid[][] = [];
    const weights: number[] = [];

    const pushList = (ids: Uuid[], weight: number) => {
      if (ids.length > 0) {
        rankedLists.push(ids);
        weights.push(weight);
      }
    };

    // lex sub-query → edge + node + episode ranked lists.
    const runLex = async (text: string, weight: number) => {
      const [edges, nodes, episodes] = await Promise.all([
        this.entityEdgeRepository.searchByFact(
          SearchByTextParamsSchema.parse({
            query: text,
            graphIds,
            limit: AGENTIC_CANDIDATE_LIMIT,
          }),
        ),
        this.entityNodeRepository.searchByName(
          SearchByTextParamsSchema.parse({
            query: text,
            graphIds,
            limit: AGENTIC_CANDIDATE_LIMIT,
          }),
          filters,
        ),
        this.episodicNodeRepository.searchByContent(
          SearchByTextParamsSchema.parse({
            query: text,
            graphIds,
            limit: AGENTIC_CANDIDATE_LIMIT,
          }),
        ),
      ]);
      for (const e of edges) edgeMap.set(e.id, e);
      for (const n of nodes) nodeMap.set(n.id, n);
      for (const ep of episodes) episodeMap.set(ep.id, ep);
      pushList(
        edges.map((e) => e.id),
        weight,
      );
      pushList(
        nodes.map((n) => n.id),
        weight,
      );
      pushList(
        episodes.map((ep) => ep.id),
        weight,
      );
    };

    // vec/hyde sub-query → edge + node ranked lists (episodes have no embedding).
    const runVec = async (embedding: number[], weight: number) => {
      const simParams = SearchBySimilarityParamsSchema.parse({
        embedding,
        graphIds,
        limit: AGENTIC_CANDIDATE_LIMIT,
        minScore: 0,
      });
      const [edges, nodes] = await Promise.all([
        this.entityEdgeRepository.searchBySimilarity(simParams, filters),
        this.entityNodeRepository.searchBySimilarity(simParams, filters),
      ]);
      for (const e of edges) edgeMap.set(e.id, e);
      for (const n of nodes) nodeMap.set(n.id, n);
      pushList(
        edges.map((e) => e.id),
        weight,
      );
      pushList(
        nodes.map((n) => n.id),
        weight,
      );
    };

    // Embed the semantic texts (original + vec/hyde sub-queries) up front.
    const semanticTexts = [
      originalQuery,
      ...queries.filter((q) => q.type !== SubQueryType.lex).map((q) => q.text),
    ];
    const embeddings = await Promise.all(
      semanticTexts.map((t) => this.embeddingService.embedText(t)),
    );
    const embeddingByText = new Map<string, number[]>();
    semanticTexts.forEach((t, i) => {
      const e = embeddings[i];
      if (e) embeddingByText.set(t, e);
    });

    const tasks: Promise<void>[] = [];
    // Original query, double-weighted, as both lex and vec.
    tasks.push(runLex(originalQuery, RRF_ORIGINAL_WEIGHT));
    const origEmbedding = embeddingByText.get(originalQuery);
    if (origEmbedding) tasks.push(runVec(origEmbedding, RRF_ORIGINAL_WEIGHT));
    // Agent-authored sub-queries.
    for (const q of queries) {
      if (q.type === SubQueryType.lex) {
        tasks.push(runLex(q.text, 1));
      } else {
        const emb = embeddingByText.get(q.text);
        if (emb) tasks.push(runVec(emb, 1));
      }
    }
    await Promise.all(tasks);

    const [fusedIds, fusedScores] = weightedRrf(rankedLists, weights);
    const scoreById = new Map(fusedIds.map((id, i) => [id, fusedScores[i]]));
    const topIds = fusedIds.slice(0, limit);

    const edges: EntityEdge[] = [];
    const nodes: EntityNode[] = [];
    const episodes: EpisodicNode[] = [];

    for (const id of topIds) {
      const edge = edgeMap.get(id);
      if (edge) {
        edges.push(edge);
        continue;
      }
      const node = nodeMap.get(id);
      if (node) {
        nodes.push(node);
        continue;
      }
      const episode = episodeMap.get(id);
      if (episode) episodes.push(episode);
    }
    const episodeSnippets = episodes.length
      ? await this.episodicNodeRepository.searchSnippets(
          episodes.map((e) => e.id),
          originalQuery,
        )
      : new Map<Uuid, string>();

    return {
      edges,
      nodes,
      episodes,
      scores: new Map(topIds.map((id) => [id, scoreById.get(id) ?? 0])),
      episodeSnippets,
    };
  }
  // =============== TODO: QMD STYLE FUNCTIONS END ================================

  @Span('search', { onResult: metricsOnResult })
  private async searchImpl(
    options: SearchOptionsInput,
  ): Promise<SearchResults & { metrics: SpanMetrics }> {
    const parsed: SearchOptions = SearchOptionsSchema.parse(options);
    const ctx: LlmContext = {
      userId: parsed.userId,
      tags: [
        'knowledge-graph',
        'retrieval',
        ...parsed.graphIds.map((id) => `group:${id}`),
      ],
      metadata: {
        query: parsed.query.slice(0, 200),
      },
    };

    const { query, graphIds, config, filters, centerNodeId, originNodeIds, userId } =
      parsed;

    const baseMetrics: SpanMetrics = {
      'user.id': ctx.userId,
      'session.id': ctx.sessionId,
      'query.length': query.length,
    };

    if (!query.trim()) {
      return {
        ...emptySearchResults(),
        metrics: {
          ...baseMetrics,
          'results.edges': 0,
          'results.nodes': 0,
          'results.episodes': 0,
          'results.communities': 0,
        },
      };
    }

    const limit = config.limit;
    const minScore = config.rerankerMinScore;

    const configs = [
      config.edgeConfig,
      config.nodeConfig,
      config.episodeConfig,
      config.communityConfig,
    ].filter(Boolean);

    const needsVector = configs.some(
      (c) =>
        c!.searchMethods.some(
          (m) =>
            m === EdgeSearchMethod.cosine_similarity ||
            m === NodeSearchMethod.cosine_similarity ||
            m === CommunitySearchMethod.cosine_similarity,
        ) ||
        c!.reranker === EdgeReranker.mmr ||
        c!.reranker === NodeReranker.mmr ||
        c!.reranker === CommunityReranker.mmr,
    );

    const needsModel = configs.some(
      (c) =>
        c!.reranker === EdgeReranker.cross_encoder ||
        c!.reranker === NodeReranker.cross_encoder ||
        c!.reranker === EpisodeReranker.cross_encoder ||
        c!.reranker === CommunityReranker.cross_encoder,
    );

    const [queryVector, model] = await Promise.all([
      needsVector ? this.embeddingService.embedText(query) : Promise.resolve(null),
      needsModel ? this.llmService.getActiveModel(userId) : Promise.resolve(null),
    ]);

    const [edgeResult, nodeResult, episodeResult, communityResult] = await Promise.all([
      config.edgeConfig
        ? this.edgeSearch(
            query,
            queryVector,
            graphIds,
            config.edgeConfig,
            filters,
            limit,
            minScore,
            model,
            centerNodeId,
            originNodeIds,
            ctx,
          )
        : Promise.resolve([[], []] as [EntityEdge[], number[]]),
      config.nodeConfig
        ? this.nodeSearch(
            query,
            queryVector,
            graphIds,
            config.nodeConfig,
            filters,
            limit,
            minScore,
            model,
            centerNodeId,
            originNodeIds,
            ctx,
          )
        : Promise.resolve([[], []] as [EntityNode[], number[]]),
      config.episodeConfig
        ? this.episodeSearch(
            query,
            graphIds,
            config.episodeConfig,
            limit,
            minScore,
            model,
            ctx,
          )
        : Promise.resolve([[], []] as [EpisodicNode[], number[]]),
      config.communityConfig
        ? this.communitySearch(
            query,
            queryVector,
            graphIds,
            config.communityConfig,
            limit,
            minScore,
            model,
            ctx,
          )
        : Promise.resolve([[], []] as [Community[], number[]]),
    ]);

    const [edges, edgeScoreArr] = edgeResult;
    const [nodes, nodeScoreArr] = nodeResult;
    const [episodes, episodeScoreArr] = episodeResult;
    const [communities, communityScoreArr] = communityResult;

    return {
      edges,
      edgeScores: new Map(edges.map((e, i) => [e.id, edgeScoreArr[i]])),
      nodes,
      nodeScores: new Map(nodes.map((n, i) => [n.id, nodeScoreArr[i]])),
      episodes,
      episodeScores: new Map(episodes.map((ep, i) => [ep.id, episodeScoreArr[i]])),
      communities,
      communityScores: new Map(communities.map((c, i) => [c.id, communityScoreArr[i]])),
      metrics: {
        ...baseMetrics,
        'results.edges': edges.length,
        'results.nodes': nodes.length,
        'results.episodes': episodes.length,
        'results.communities': communities.length,
      },
    };
  }

  // ─── Edge search ───────────────────────────────────────────────────────────

  private async edgeSearch(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: EdgeSearchConfig,
    filters: SearchFilters | undefined,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    centerNodeId?: Uuid,
    originNodeIds?: Uuid[],
    ctx?: LlmContext,
  ): Promise<[EntityEdge[], number[]]> {
    const { edges, scores } = await this.edgeSearchImpl(
      query,
      queryVector,
      graphIds,
      config,
      filters,
      limit,
      minScore,
      model,
      centerNodeId,
      originNodeIds,
      ctx,
    );
    return [edges, scores];
  }

  @Span('search.edge', { observationKind: 'retriever', onResult: metricsOnResult })
  private async edgeSearchImpl(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: EdgeSearchConfig,
    filters: SearchFilters | undefined,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    centerNodeId?: Uuid,
    originNodeIds?: Uuid[],
    ctx?: LlmContext,
  ): Promise<{ edges: EntityEdge[]; scores: number[]; metrics: SpanMetrics }> {
    const fetch = 2 * limit;
    const edgeMap = new Map<Uuid, EntityEdge>();

    const bm25Ids: Uuid[] = [];
    const cosineIds: Uuid[] = [];
    let bfsIds: Uuid[] = [];

    const tasks: Promise<void>[] = [];

    if (config.searchMethods.includes(EdgeSearchMethod.bm25)) {
      tasks.push(
        this.entityEdgeRepository
          .searchByFact(SearchByTextParamsSchema.parse({ query, graphIds, limit: fetch }))
          .then((edges) => {
            for (const e of edges) edgeMap.set(e.id, e);
            bm25Ids.push(...edges.map((e) => e.id));
          }),
      );
    }

    if (
      config.searchMethods.includes(EdgeSearchMethod.cosine_similarity) &&
      queryVector
    ) {
      tasks.push(
        this.entityEdgeRepository
          .searchBySimilarity(
            SearchBySimilarityParamsSchema.parse({
              embedding: queryVector,
              graphIds,
              limit: fetch,
              minScore: config.simMinScore,
            }),
            filters,
          )
          .then((edges) => {
            for (const e of edges) edgeMap.set(e.id, e);
            cosineIds.push(...edges.map((e) => e.id));
          }),
      );
    }

    await Promise.all(tasks);

    if (config.searchMethods.includes(EdgeSearchMethod.bfs)) {
      const origins =
        originNodeIds && originNodeIds.length > 0
          ? originNodeIds
          : [...edgeMap.values()].map((e) => e.sourceNodeId);

      if (origins.length > 0) {
        const bfsEdges = await this.entityEdgeRepository.searchByBfs(
          SearchByBfsParamsSchema.parse({
            originNodeIds: origins,
            graphIds,
            limit: fetch,
            maxDepth: config.maxDepth,
          }),
          filters,
        );
        for (const e of bfsEdges) edgeMap.set(e.id, e);
        bfsIds = bfsEdges.map((e) => e.id);
      }
    }

    const reranker = config.reranker;
    const rerankerMin = config.rerankerMinScore ?? minScore;

    let rankedIds: Uuid[];
    let rankedScores: number[];

    if (reranker === EdgeReranker.rrf) {
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    } else if (reranker === EdgeReranker.mmr && queryVector) {
      const vectorPairs = new Map<Uuid, number[]>();
      for (const [id, edge] of edgeMap) {
        if (edge.factEmbedding) vectorPairs.set(id, edge.factEmbedding);
      }
      [rankedIds, rankedScores] = mmr(
        queryVector,
        vectorPairs,
        config.mmrLambda,
        rerankerMin,
      );
    } else if (reranker === EdgeReranker.cross_encoder && model) {
      const candidates = [...edgeMap.values()].slice(0, limit);
      [rankedIds, rankedScores] = await crossEncoderReranker(
        model,
        query,
        candidates.map((e) => ({ id: e.id, text: e.fact })),
        rerankerMin,
        { llmTracer: this.llmTracer, ctx },
      );
    } else if (reranker === EdgeReranker.node_distance) {
      if (!centerNodeId) {
        throw new Error('centerNodeId is required for node_distance reranker');
      }
      const sourceIds = [...new Set([...edgeMap.values()].map((e) => e.sourceNodeId))];
      const [rankedSourceIds, sourceScores] = await nodeDistanceReranker(
        this.entityNodeRepository,
        sourceIds,
        centerNodeId,
        rerankerMin,
      );
      // Map source IDs back to edge IDs (preserving order)
      const sourceScoreMap = new Map(rankedSourceIds.map((u, i) => [u, sourceScores[i]]));
      const edgeEntries: [Uuid, number][] = [];
      for (const [id, edge] of edgeMap) {
        const score = sourceScoreMap.get(edge.sourceNodeId);
        if (score !== undefined) edgeEntries.push([id, score]);
      }
      edgeEntries.sort((a, b) => b[1] - a[1]);
      rankedIds = edgeEntries.map(([u]) => u);
      rankedScores = edgeEntries.map(([, s]) => s);
    } else if (reranker === EdgeReranker.episode_mentions) {
      // RRF preliminary ranking, then sort by episode count descending.
      // Matches Python search.py:256-305: rrf runs first for both EdgeReranker.rrf
      // and EdgeReranker.episode_mentions, then episode_mentions applies the sort.
      const [rrfIds, rrfScores] = rrf(
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
      const rrfScoreMap = new Map(rrfIds.map((u, i) => [u, rrfScores[i]]));
      const rrfEdges = rrfIds
        .map((id) => edgeMap.get(id))
        .filter((e): e is EntityEdge => e !== undefined);
      rrfEdges.sort((a, b) => b.episodes.length - a.episodes.length);
      rankedIds = rrfEdges.map((e) => e.id);
      rankedScores = rankedIds.map((u) => rrfScoreMap.get(u) ?? 0);
    } else {
      // Fallback: RRF
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    }

    const resultEdges = rankedIds
      .slice(0, limit)
      .map((id) => edgeMap.get(id))
      .filter((e): e is EntityEdge => e !== undefined);
    const resultScores = rankedScores.slice(0, limit).slice(0, resultEdges.length);

    return {
      edges: resultEdges,
      scores: resultScores,
      metrics: {
        'query.length': query.length,
        limit: limit,
        minScore: minScore,
        'config.searchMethods': config.searchMethods.join(','),
        'config.reranker': config.reranker,
        'result.count': resultEdges.length,
      },
    };
  }

  // ─── Node search ───────────────────────────────────────────────────────────

  private async nodeSearch(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: NodeSearchConfig,
    filters: SearchFilters | undefined,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    centerNodeId?: Uuid,
    originNodeIds?: Uuid[],
    ctx?: LlmContext,
  ): Promise<[EntityNode[], number[]]> {
    const { nodes, scores } = await this.nodeSearchImpl(
      query,
      queryVector,
      graphIds,
      config,
      filters,
      limit,
      minScore,
      model,
      centerNodeId,
      originNodeIds,
      ctx,
    );
    return [nodes, scores];
  }

  @Span('search.node', { observationKind: 'retriever', onResult: metricsOnResult })
  private async nodeSearchImpl(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: NodeSearchConfig,
    filters: SearchFilters | undefined,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    centerNodeId?: Uuid,
    originNodeIds?: Uuid[],
    ctx?: LlmContext,
  ): Promise<{ nodes: EntityNode[]; scores: number[]; metrics: SpanMetrics }> {
    const fetch = 2 * limit;
    const nodeMap = new Map<Uuid, EntityNode>();

    const bm25Ids: Uuid[] = [];
    const cosineIds: Uuid[] = [];
    let bfsIds: Uuid[] = [];

    const tasks: Promise<void>[] = [];

    if (config.searchMethods.includes(NodeSearchMethod.bm25)) {
      tasks.push(
        this.entityNodeRepository
          .searchByName(
            SearchByTextParamsSchema.parse({ query, graphIds, limit: fetch }),
            filters,
          )
          .then((nodes) => {
            for (const n of nodes) nodeMap.set(n.id, n);
            bm25Ids.push(...nodes.map((n) => n.id));
          }),
      );
    }

    if (
      config.searchMethods.includes(NodeSearchMethod.cosine_similarity) &&
      queryVector
    ) {
      tasks.push(
        this.entityNodeRepository
          .searchBySimilarity(
            SearchBySimilarityParamsSchema.parse({
              embedding: queryVector,
              graphIds,
              limit: fetch,
              minScore: config.simMinScore,
            }),
            filters,
          )
          .then((nodes) => {
            for (const n of nodes) nodeMap.set(n.id, n);
            cosineIds.push(...nodes.map((n) => n.id));
          }),
      );
    }

    await Promise.all(tasks);

    if (config.searchMethods.includes(NodeSearchMethod.bfs)) {
      const origins =
        originNodeIds && originNodeIds.length > 0 ? originNodeIds : [...nodeMap.keys()];

      if (origins.length > 0) {
        const bfsNodes = await this.entityNodeRepository.searchByBfs(
          SearchByBfsParamsSchema.parse({
            originNodeIds: origins,
            graphIds,
            limit: fetch,
            maxDepth: config.maxDepth,
          }),
          filters,
        );
        for (const n of bfsNodes) nodeMap.set(n.id, n);
        bfsIds = bfsNodes.map((n) => n.id);
      }
    }

    const reranker = config.reranker;
    const rerankerMin = config.rerankerMinScore ?? minScore;

    let rankedIds: Uuid[];
    let rankedScores: number[];

    if (reranker === NodeReranker.rrf) {
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    } else if (reranker === NodeReranker.mmr && queryVector) {
      const vectorPairs = new Map<Uuid, number[]>();
      for (const [id, node] of nodeMap) {
        if (node.nameEmbedding) vectorPairs.set(id, node.nameEmbedding);
      }
      [rankedIds, rankedScores] = mmr(
        queryVector,
        vectorPairs,
        config.mmrLambda,
        rerankerMin,
      );
    } else if (reranker === NodeReranker.cross_encoder && model) {
      const candidates = [...nodeMap.values()].slice(0, limit);
      [rankedIds, rankedScores] = await crossEncoderReranker(
        model,
        query,
        candidates.map((n) => ({ id: n.id, text: n.name })),
        rerankerMin,
        { llmTracer: this.llmTracer, ctx },
      );
    } else if (reranker === NodeReranker.node_distance) {
      if (!centerNodeId) {
        throw new Error('centerNodeId is required for node_distance reranker');
      }
      [rankedIds, rankedScores] = await nodeDistanceReranker(
        this.entityNodeRepository,
        [...nodeMap.keys()],
        centerNodeId,
        rerankerMin,
      );
    } else if (reranker === NodeReranker.episode_mentions) {
      [rankedIds, rankedScores] = await episodeMentionsReranker(
        this.entityNodeRepository,
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    } else {
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds, bfsIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    }

    const resultNodes = rankedIds
      .slice(0, limit)
      .map((id) => nodeMap.get(id))
      .filter((n): n is EntityNode => n !== undefined);
    const resultScores = rankedScores.slice(0, limit).slice(0, resultNodes.length);

    return {
      nodes: resultNodes,
      scores: resultScores,
      metrics: {
        'query.length': query.length,
        limit: limit,
        minScore: minScore,
        'config.searchMethods': config.searchMethods.join(','),
        'config.reranker': config.reranker,
        'result.count': resultNodes.length,
      },
    };
  }

  // ─── Episode search ─────────────────────────────────────────────────────────

  private async episodeSearch(
    query: string,
    graphIds: Uuid[],
    config: EpisodeSearchConfig,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    ctx?: LlmContext,
  ): Promise<[EpisodicNode[], number[]]> {
    const { episodes, scores } = await this.episodeSearchImpl(
      query,
      graphIds,
      config,
      limit,
      minScore,
      model,
      ctx,
    );
    return [episodes, scores];
  }

  @Span('search.episode', { observationKind: 'retriever', onResult: metricsOnResult })
  private async episodeSearchImpl(
    query: string,
    graphIds: Uuid[],
    config: EpisodeSearchConfig,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    ctx?: LlmContext,
  ): Promise<{ episodes: EpisodicNode[]; scores: number[]; metrics: SpanMetrics }> {
    const fetch = 2 * limit;
    const rerankerMin = config.rerankerMinScore ?? minScore;

    const bm25Episodes = config.searchMethods.includes(EpisodeSearchMethod.bm25)
      ? await this.episodicNodeRepository.searchByContent(
          SearchByTextParamsSchema.parse({ query, graphIds, limit: fetch }),
        )
      : [];

    const episodeMap = new Map(bm25Episodes.map((ep) => [ep.id, ep]));
    const bm25Ids = bm25Episodes.map((ep) => ep.id);

    let rankedIds: Uuid[];
    let rankedScores: number[];

    if (config.reranker === EpisodeReranker.cross_encoder && model) {
      [rankedIds, rankedScores] = await crossEncoderReranker(
        model,
        query,
        bm25Episodes.slice(0, limit).map((ep) => ({ id: ep.id, text: ep.content })),
        rerankerMin,
        { llmTracer: this.llmTracer, ctx },
      );
    } else {
      [rankedIds, rankedScores] = rrf([bm25Ids], rerankerMin);
    }

    const resultEpisodes = rankedIds
      .slice(0, limit)
      .map((id) => episodeMap.get(id))
      .filter((ep): ep is EpisodicNode => ep !== undefined);
    const resultScores = rankedScores.slice(0, limit).slice(0, resultEpisodes.length);

    return {
      episodes: resultEpisodes,
      scores: resultScores,
      metrics: {
        'query.length': query.length,
        limit: limit,
        minScore: minScore,
        'result.count': resultEpisodes.length,
      },
    };
  }

  // ─── Community search ───────────────────────────────────────────────────────

  private async communitySearch(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: CommunitySearchConfig,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    ctx?: LlmContext,
  ): Promise<[Community[], number[]]> {
    const { communities, scores } = await this.communitySearchImpl(
      query,
      queryVector,
      graphIds,
      config,
      limit,
      minScore,
      model,
      ctx,
    );
    return [communities, scores];
  }

  @Span('search.community', { observationKind: 'retriever', onResult: metricsOnResult })
  private async communitySearchImpl(
    query: string,
    queryVector: number[] | null,
    graphIds: Uuid[],
    config: CommunitySearchConfig,
    limit: number,
    minScore: number,
    model: BaseChatModel | null,
    ctx?: LlmContext,
  ): Promise<{ communities: Community[]; scores: number[]; metrics: SpanMetrics }> {
    const fetch = 2 * limit;
    const communityMap = new Map<Uuid, Community>();
    const rerankerMin = config.rerankerMinScore ?? minScore;

    const bm25Ids: Uuid[] = [];
    const cosineIds: Uuid[] = [];

    const tasks: Promise<void>[] = [];

    if (config.searchMethods.includes(CommunitySearchMethod.bm25)) {
      tasks.push(
        this.communityRepository
          .searchByName(SearchByTextParamsSchema.parse({ query, graphIds, limit: fetch }))
          .then((nodes) => {
            for (const n of nodes) communityMap.set(n.id, n);
            bm25Ids.push(...nodes.map((n) => n.id));
          }),
      );
    }

    if (
      config.searchMethods.includes(CommunitySearchMethod.cosine_similarity) &&
      queryVector
    ) {
      tasks.push(
        this.communityRepository
          .searchBySimilarity(
            SearchBySimilarityParamsSchema.parse({
              embedding: queryVector,
              graphIds,
              limit: fetch,
              minScore: config.simMinScore,
            }),
          )
          .then((nodes) => {
            for (const n of nodes) communityMap.set(n.id, n);
            cosineIds.push(...nodes.map((n) => n.id));
          }),
      );
    }

    await Promise.all(tasks);

    let rankedIds: Uuid[];
    let rankedScores: number[];

    if (config.reranker === CommunityReranker.rrf) {
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    } else if (config.reranker === CommunityReranker.mmr && queryVector) {
      const vectorPairs = new Map<Uuid, number[]>();
      for (const [id, community] of communityMap) {
        if (community.nameEmbedding) vectorPairs.set(id, community.nameEmbedding);
      }
      [rankedIds, rankedScores] = mmr(
        queryVector,
        vectorPairs,
        config.mmrLambda,
        rerankerMin,
      );
    } else if (config.reranker === CommunityReranker.cross_encoder && model) {
      const candidates = [...communityMap.values()].slice(0, limit);
      [rankedIds, rankedScores] = await crossEncoderReranker(
        model,
        query,
        candidates.map((c) => ({ id: c.id, text: c.name })),
        rerankerMin,
        { llmTracer: this.llmTracer, ctx },
      );
    } else {
      [rankedIds, rankedScores] = rrf(
        [bm25Ids, cosineIds].filter((l) => l.length > 0),
        rerankerMin,
      );
    }

    const resultCommunities = rankedIds
      .slice(0, limit)
      .map((id) => communityMap.get(id))
      .filter((c): c is Community => c !== undefined);
    const resultScores = rankedScores.slice(0, limit).slice(0, resultCommunities.length);

    return {
      communities: resultCommunities,
      scores: resultScores,
      metrics: {
        'query.length': query.length,
        limit: limit,
        minScore: minScore,
        'config.searchMethods': config.searchMethods.join(','),
        'config.reranker': config.reranker,
        'result.count': resultCommunities.length,
      },
    };
  }
}
