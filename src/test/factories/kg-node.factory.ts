import {
  Community,
  createCommunity,
  createEntityNode,
  createEpisodicNode,
  createSaga,
  EntityNode,
  EpisodeType,
  EpisodicNode,
  Saga,
} from '@/knowledge-graph/models';
import {
  NodeLabelSchema,
  NodeLabelsSchema,
  NodeNameSchema,
} from '@/knowledge-graph/types';

import { KG_REFERENCE_TIME, KG_TEST_GRAPH_ID } from './kg-constants';

type WithStrings<T> = Omit<Partial<T>, 'name' | 'labels'> & {
  name?: string;
  labels?: string[];
};

export class KgNodeFactory {
  static createEntityNode(opts: WithStrings<EntityNode> = {}): EntityNode {
    const { name, labels, ...rest } = opts;
    return createEntityNode({
      name: NodeNameSchema.parse(name ?? 'TestEntity'),
      labels: labels ? NodeLabelsSchema.parse(labels) : [NodeLabelSchema.parse('Entity')],
      graphId: KG_TEST_GRAPH_ID,
      ...rest,
    });
  }

  static createEpisodicNode(opts: WithStrings<EpisodicNode> = {}): EpisodicNode {
    const { name, labels, ...rest } = opts;
    return createEpisodicNode({
      name: NodeNameSchema.parse(name ?? 'Test Episode'),
      graphId: KG_TEST_GRAPH_ID,
      content: 'test content',
      validAt: KG_REFERENCE_TIME,
      source: EpisodeType.text,
      sourceDescription: 'test',
      labels: labels
        ? NodeLabelsSchema.parse(labels)
        : [NodeLabelSchema.parse('Episodic')],
      ...rest,
    });
  }

  static createCommunity(
    opts: Partial<Omit<Community, 'name'>> & { name?: string } = {},
  ): Community {
    const { name, ...rest } = opts;
    return createCommunity({
      name: NodeNameSchema.parse(name ?? 'Test Community'),
      graphId: KG_TEST_GRAPH_ID,
      memberIds: [],
      ...rest,
    });
  }

  static createSaga(opts: WithStrings<Saga> = {}): Saga {
    const { name, labels, ...rest } = opts;
    return createSaga({
      name: NodeNameSchema.parse(name ?? 'Test Saga'),
      graphId: KG_TEST_GRAPH_ID,
      labels: labels ? NodeLabelsSchema.parse(labels) : [NodeLabelSchema.parse('Saga')],
      ...rest,
    });
  }
}
