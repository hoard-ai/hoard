// We use @google/genai instead of @langchain/google-genai (which the LLM
// factory still uses for chat) because LangChain's GoogleGenerativeAIEmbeddings
// wrapper doesn't expose `outputDimensionality` as of now. We need it to truncate to the
// Postgres `vector(768)` schema and to opt into gemini-embedding-2's
// server-side L2 normalization on truncated outputs.
import { GoogleGenAI } from '@google/genai';
import { Injectable } from '@nestjs/common';

import { EmbeddingConfigService } from '@/config/embedding';
import {
  metricsOnResult,
  recordSpanEvent,
  Span,
  type SpanMetrics,
} from '@/observability';
import { RateLimiterService } from '@/providers/rate-limit';

import { EntityEdge, EntityNode } from '../models';

const RATE_LIMIT_KEY = 'embedding';

@Injectable()
export class EmbeddingService {
  private readonly client: GoogleGenAI | null;
  private readonly modelName: string;
  private readonly _dimensions: number;
  private readonly concurrencyLimit: number;

  constructor(
    embeddingConfig: EmbeddingConfigService,
    private readonly rateLimiter: RateLimiterService,
  ) {
    this._dimensions = embeddingConfig.dimensions;
    this.modelName = embeddingConfig.googleModel;
    this.concurrencyLimit = embeddingConfig.concurrencyLimit;

    if (!embeddingConfig.embeddingEnabled) {
      this.client = null;
      return;
    }
    const apiKey = embeddingConfig.googleApiKey;
    if (!apiKey) {
      throw new Error(
        'GOOGLE_EMBEDDING_API_KEY is required when EMBEDDING_ENABLED is true',
      );
    }
    this.client = new GoogleGenAI({ apiKey });
  }

  /** Output dimension of the configured embedding model. */
  get dimensions(): number {
    return this._dimensions;
  }

  // TODO: examine the provider's per-request contents cap. A single call with a
  // very large `texts` array may exceed it
  private async embed(texts: string[]): Promise<number[][]> {
    return this.rateLimiter.schedule(
      RATE_LIMIT_KEY,
      this.concurrencyLimit,
      undefined,
      async () => {
        const res = await this.client!.models.embedContent({
          model: this.modelName,
          // Wrap each text as its own Content so the SDK returns N embeddings,
          // not a single embedding for N concatenated parts.
          contents: texts.map((text) => ({ parts: [{ text }] })),
          config: { outputDimensionality: this._dimensions },
        });
        const embeddings = res.embeddings;
        if (!embeddings || embeddings.length !== texts.length) {
          throw new Error(
            `Embedding API returned ${embeddings?.length ?? 0} vectors for ${texts.length} inputs`,
          );
        }
        return embeddings.map((e, i) => {
          if (!e.values) {
            throw new Error(`Embedding API returned no values for input index ${i}`);
          }
          return e.values;
        });
      },
      (waitMs) =>
        recordSpanEvent('ratelimit.admitted', {
          'ratelimit.key': RATE_LIMIT_KEY,
          'ratelimit.queue_wait_ms': waitMs,
        }),
    );
  }

  async embedNodes(nodes: EntityNode[]): Promise<EntityNode[]> {
    const { nodes: out } = await this.embedNodesImpl(nodes);
    return out;
  }

  @Span('embed.nodes', { observationKind: 'embedding', onResult: metricsOnResult })
  private async embedNodesImpl(
    nodes: EntityNode[],
  ): Promise<{ nodes: EntityNode[]; metrics: SpanMetrics }> {
    if (this.client === null) {
      return { nodes, metrics: { count: 0, model: this.modelName, skipped: true } };
    }

    const toEmbed = nodes.filter((n) => n.nameEmbedding === null);
    if (toEmbed.length === 0) {
      return { nodes, metrics: { count: 0, model: this.modelName } };
    }

    const vectors = await this.embed(toEmbed.map((n) => n.name));
    toEmbed.forEach((n, i) => {
      n.nameEmbedding = vectors[i];
    });
    return { nodes, metrics: { count: toEmbed.length, model: this.modelName } };
  }

  async embedText(text: string): Promise<number[] | null> {
    const { vector } = await this.embedTextImpl(text);
    return vector;
  }

  @Span('embed.text', { observationKind: 'embedding', onResult: metricsOnResult })
  private async embedTextImpl(
    text: string,
  ): Promise<{ vector: number[] | null; metrics: SpanMetrics }> {
    if (this.client === null) {
      return { vector: null, metrics: { model: this.modelName, skipped: true } };
    }
    const [vector] = await this.embed([text]);
    return { vector, metrics: { model: this.modelName } };
  }

  async embedEdges(edges: EntityEdge[]): Promise<EntityEdge[]> {
    const { edges: out } = await this.embedEdgesImpl(edges);
    return out;
  }

  @Span('embed.edges', { observationKind: 'embedding', onResult: metricsOnResult })
  private async embedEdgesImpl(
    edges: EntityEdge[],
  ): Promise<{ edges: EntityEdge[]; metrics: SpanMetrics }> {
    if (this.client === null) {
      return { edges, metrics: { count: 0, model: this.modelName, skipped: true } };
    }

    const toEmbed = edges.filter((e) => e.factEmbedding === null);
    if (toEmbed.length === 0) {
      return { edges, metrics: { count: 0, model: this.modelName } };
    }

    const vectors = await this.embed(toEmbed.map((e) => e.fact));
    toEmbed.forEach((e, i) => {
      e.factEmbedding = vectors[i];
    });
    return { edges, metrics: { count: toEmbed.length, model: this.modelName } };
  }
}
