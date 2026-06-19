import {
  createEntityEdge,
  createEpisodicEdge,
  EntityEdge,
  EpisodicEdge,
} from '@/knowledge-graph/models';
import { RelationshipTypeSchema } from '@/knowledge-graph/types';

import { KG_REFERENCE_TIME, KG_TEST_GRAPH_ID, kgId } from './kg-constants';

type WithStringName<T> = Omit<Partial<T>, 'name'> & { name?: string };

export class KgEdgeFactory {
  static createEntityEdge(opts: WithStringName<EntityEdge> = {}): EntityEdge {
    const { name, ...rest } = opts;
    return createEntityEdge({
      name: RelationshipTypeSchema.parse(name ?? 'RELATED_TO'),
      fact: 'A is related to B',
      graphId: KG_TEST_GRAPH_ID,
      sourceNodeId: kgId(),
      targetNodeId: kgId(),
      validAt: KG_REFERENCE_TIME,
      ...rest,
    });
  }

  static createEpisodicEdge(opts: Partial<EpisodicEdge> = {}): EpisodicEdge {
    return createEpisodicEdge({
      graphId: KG_TEST_GRAPH_ID,
      sourceNodeId: kgId(),
      targetNodeId: kgId(),
      ...opts,
    });
  }
}
