import { Uuid } from '@/common/schemas';

import { EntityEdge, EntityNode, EpisodicNode } from '../../models';

// TODO: fixed size assumes small chunks - make adaptive on the summary prompt's
// token budget (the episode-text chunk union + node payload) so large chunks
// don't overflow the context.
export const MAX_NODES_PER_SUMMARY_BATCH = 30;

export type ExtractNodesResult = {
  nodes: EntityNode[];
  // Chunk indices each node was extracted from (unioned across chunks).
  chunkIndicesByNodeId: Map<Uuid, Set<number>>;
};

export type NodeEpisodeContext = Map<
  Uuid,
  {
    episode: EpisodicNode;
    previousEpisodes: EpisodicNode[];
    // The episode's chunks + this node's chunk indices, so attribute/summary
    // episode text is scoped to where the node was discussed (selectChunkText).
    chunks: string[];
    sourceChunkIndices: Set<number>;
  }
>;

export type ExtractEdgesResult = {
  edges: EntityEdge[];
  // Chunk index each edge was extracted from (stays a singleton - dedup keeps
  // each edge id's own origin entry and never unions, see EdgeChunkSources).
  chunkIndicesByEdgeId: Map<Uuid, Set<number>>;
};
