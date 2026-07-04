import type { SearchConfigInput } from './types';
import {
  CommunityReranker,
  CommunitySearchMethod,
  DEFAULT_MIN_SCORE,
  DEFAULT_SEARCH_LIMIT,
  EdgeReranker,
  EdgeSearchMethod,
  EpisodeReranker,
  EpisodeSearchMethod,
  NodeReranker,
  NodeSearchMethod,
} from './types';

// NOTE: None of these recipes include temporal filters, matching the Python
// Graphiti source (search_config_recipes.py). Temporal filters (e.g. excluding
// expired_at IS NOT NULL edges) must be added by callers via SearchConfig.filters.

// ─── Combined (all entity types) ────────────────────────────────────────────

export const COMBINED_HYBRID_SEARCH_RRF: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.rrf,
  },
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.rrf,
  },
  episodeConfig: {
    searchMethods: [EpisodeSearchMethod.bm25],
    reranker: EpisodeReranker.rrf,
  },
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.rrf,
  },
};

export const COMBINED_HYBRID_SEARCH_MMR: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.mmr,
  },
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.mmr,
  },
  episodeConfig: {
    searchMethods: [EpisodeSearchMethod.bm25],
    reranker: EpisodeReranker.rrf,
  },
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.mmr,
  },
};

export const COMBINED_HYBRID_SEARCH_CROSS_ENCODER: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [
      EdgeSearchMethod.bm25,
      EdgeSearchMethod.cosine_similarity,
      EdgeSearchMethod.bfs,
    ],
    reranker: EdgeReranker.cross_encoder,
  },
  nodeConfig: {
    searchMethods: [
      NodeSearchMethod.bm25,
      NodeSearchMethod.cosine_similarity,
      NodeSearchMethod.bfs,
    ],
    reranker: NodeReranker.cross_encoder,
  },
  episodeConfig: {
    searchMethods: [EpisodeSearchMethod.bm25],
    reranker: EpisodeReranker.cross_encoder,
  },
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.cross_encoder,
  },
};

// ─── Edge-only ───────────────────────────────────────────────────────────────

export const EDGE_HYBRID_SEARCH_RRF: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.rrf,
  },
};

export const EDGE_HYBRID_SEARCH_MMR: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.mmr,
  },
};

export const EDGE_HYBRID_SEARCH_CROSS_ENCODER: SearchConfigInput = {
  limit: 10,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [
      EdgeSearchMethod.bm25,
      EdgeSearchMethod.cosine_similarity,
      EdgeSearchMethod.bfs,
    ],
    reranker: EdgeReranker.cross_encoder,
  },
};

export const EDGE_HYBRID_SEARCH_NODE_DISTANCE: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.node_distance,
  },
};

export const EDGE_HYBRID_SEARCH_EPISODE_MENTIONS: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  edgeConfig: {
    searchMethods: [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
    reranker: EdgeReranker.episode_mentions,
  },
};

// ─── Node-only ───────────────────────────────────────────────────────────────

export const NODE_HYBRID_SEARCH_RRF: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.rrf,
  },
};

export const NODE_HYBRID_SEARCH_MMR: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.mmr,
  },
};

export const NODE_HYBRID_SEARCH_CROSS_ENCODER: SearchConfigInput = {
  limit: 10,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  nodeConfig: {
    searchMethods: [
      NodeSearchMethod.bm25,
      NodeSearchMethod.cosine_similarity,
      NodeSearchMethod.bfs,
    ],
    reranker: NodeReranker.cross_encoder,
  },
};

export const NODE_HYBRID_SEARCH_NODE_DISTANCE: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.node_distance,
  },
};

export const NODE_HYBRID_SEARCH_EPISODE_MENTIONS: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  nodeConfig: {
    searchMethods: [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
    reranker: NodeReranker.episode_mentions,
  },
};

// ─── Community-only ──────────────────────────────────────────────────────────

export const COMMUNITY_HYBRID_SEARCH_RRF: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.rrf,
  },
};

export const COMMUNITY_HYBRID_SEARCH_MMR: SearchConfigInput = {
  limit: DEFAULT_SEARCH_LIMIT,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.mmr,
  },
};

export const COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER: SearchConfigInput = {
  limit: 3,
  rerankerMinScore: DEFAULT_MIN_SCORE,
  communityConfig: {
    searchMethods: [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
    reranker: CommunityReranker.cross_encoder,
  },
};
