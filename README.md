# Hoard API

NestJS backend for Hoard - a provider-agnostic, graph-enabled, time-aware memory layer that any AI agent can plug into. RememberNow is one consumer of Hoard.

## Project setup

Install dependencies:

```bash
npm install
```

Fill out `.env` file (`GEMINI_API_KEY` is required):

```bash
cp .env.example .env
```

Make sure Docker is running:

```bash
docker info
```

## Stack modes

| Command                                        | What runs                                                                |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `docker compose up -d`                         | Everything except Langfuse observability                                 |
| `docker compose --profile observability up -d` | Everything, including observability                                      |
| `npm run infra:dev:up`                         | Infra + observability (no app container, since dev runs it on host)      |
| `npm run infra:test:up`                        | App deps only, no observability, uses `.env.test` (change default ports) |

## Compile and run the project

### Container

Run app in container (easiest):

```bash
docker compose up -d
```

### Dev (host)

Run app on host.

Also brings up Langfuse + MinIO + ClickHouse:

```bash
npm run prisma:generate && npm run infra:dev:up && npm run prisma:dev:deploy && npm run start:dev
```

When done developing for the day (preserves data):

```bash
npm run infra:dev:stop
```

Resume development with existing data:

```bash
npm run infra:dev:start && npm run start:dev
```

After editing `prisma/schema.prisma` (regenerate client + create a new migration):

```bash
npm run prisma:generate && npm run prisma:dev:migrate
```

Full reset + infra restart (destroys all data):

```bash
npm run infra:dev:reset
```

Or remove all data and containers manually.

```bash
npm run infra:dev:rm
```

Once dev infra is up, the following web UIs are available:

- Swagger UI: <http://localhost:3333/api> (requires the app to be running)
- Prisma Studio: <http://localhost:5555> (requires `npm run prisma:studio`)
- Langfuse: <http://localhost:3334>
- MinIO console: <http://localhost:9091>

## Development

### Pre-commit hooks

Installed automatically by `npm install` via Husky. On each commit, `lint-staged` runs ESLint (zero warnings allowed) and Prettier across staged TS/JS files, and Prettier across staged JSON/Markdown/YAML.

### Prisma conventions

The Prisma client runs with the `strictUndefinedChecks` preview feature enabled. This means passing `undefined` to any field in `data`, `where`, `take`, `orderBy`, etc. throws at runtime instead of being silently ignored.

When you have a possibly-undefined value, guard at the call site - either use a conditional spread or `Prisma.skip`.

### Frontend Type Generation

The `openapi.json` is auto-generated when you run `npm run start:dev`.
Make sure to commit changes when you modify routes/DTOs.

### Run tests

Unit tests:

```bash
npm run test
```

E2E tests (`.env.test` file must be present. Change ports to not conflict with dev stack):

```bash
npm run test:e2e
```

Test coverage:

```bash
npm run test:cov
```

### Test stability verification

The anti-flaky test runner executes unit tests once, then runs e2e tests repeatedly to detect intermittent failures and race conditions. Useful for validating test reliability after infrastructure changes.

Run with default 15 iterations:

```bash
./anti-flaky-test.sh
```

Run with custom iteration count:

```bash
./anti-flaky-test.sh 30
```

The runner will exit immediately on first failure and display the failing test output. Test infrastructure is set up once and torn down automatically on completion or interruption.

### Langfuse observability (optional)

Langfuse captures full LLM traces - including prompts and outputs - and is only intended for use in development. It is **off by default** and never enabled on the hosted product.

To turn it on for local dev:

1. Bring up the observability profile (`npm run infra:dev:up` already does this).
2. Open <http://localhost:3334> and create an account + project (any name).
3. In the project, go to **Settings -> API Keys** and create a new key pair.
4. Add the resulting keys to your `.env`:
   ```
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
5. Restart `npm run start:dev`. Traces should start appearing under the project's **Traces** tab.

When running the app inside the docker network instead of on the host, also set `LANGFUSE_BASE_URL=http://langfuse-web:3000` (the in-network hostname). The default in `.env.example` points at the host-mapped port `http://localhost:3334`.

## Concurrency & rate limiting

The ingestion pipeline touches rate-limited provider APIs (chat + embeddings) and
builds large prompts, so concurrency is bounded on three distinct axes.

### 1. Provider rate limit - `RateLimiterService`

A distributed, Redis-backed semaphore (`src/providers/rate-limit`) bounds
concurrent in-flight provider calls **per credential (token)**, across worker processes,
with crash-safe leases.

- **Chat models** are gated at the model's generation function; the LLM service
  returns a rate-limited model by default. Each pool is keyed by the credential
  owner (a user's own key, or the shared platform key), with the limit from that
  credential's config.
- **Embeddings** don't go through a chat model, so the embedding service gates
  its own provider calls through a dedicated pool.

**Quirks:**

- The **interactive agent** bypasses the limiter so a live conversation never
  queues behind background ingestion.
- The shared **platform** pool is always rate limited and cannot be bypassed.
- Cancellation rides on the model: an ingestion-scoped `AbortSignal` rejects
  queued acquires and cancels in-flight requests.

### 2. Memory backpressure - `withConcurrency` (`MEMORY_BACKPRESSURE_CONCURRENCY_LIMIT`)

`withConcurrency` (`batch-utils`) is not a rate limit - it caps how many
large prompts are materialized at once, per fan-out. It survives only where a
fan-out builds ~MB-scale prompts (previous-episodes context is currently
untruncated) and isn't already bounded by a caller-level per-episode cap:

### 3. Worker job parallelism (BullMQ `@Processor` concurrency)

How many jobs a queue worker runs at once - orthogonal to the above (each job
still draws on the semaphores internally).

## Acknowledgements

The knowledge graph pipeline in this project is a modified TypeScript port of [Graphiti](https://github.com/getzep/graphiti) by Zep AI, adapted for this codebase for the following reasons:

- I need to implement a Graph Versioning system to prevent data loss due to agent misbehavior, which is impossible without direct code access.
- Graphiti couples tightly to specific providers. This port integrates with [LangChain](https://js.langchain.com/) so any supported model can be swapped in without changing pipeline code.
- Graphiti maintains provider abstractions for its graph layer. Managed to make Graphiti work with just Postgres ([pgvector](https://github.com/pgvector/pgvector) + [pgvectorscale](https://github.com/timescale/pgvectorscale) for ANN search), so those abstractions are unnecessary and removing them adds room for enhancements.
- Hoard is privacy-focused software. Depending on upstream code that cannot be audited, patched, or controlled introduces risk.

Approximate nearest-neighbor vector search is powered by [pgvectorscale](https://github.com/timescale/pgvectorscale) (PostgreSQL License) from Tiger Data, layered on top of [pgvector](https://github.com/pgvector/pgvector). The knowledge-graph migration enables the `vectorscale` extension and builds StreamingDiskANN indexes (with SBQ compression and label-based filtering) over the knowledge graph's vector embeddings - see `prisma/migrations/20260517000000_knowledge_graph/migration.sql`.

The `@Span` and `@Traceable` decorators in `src/observability/decorators/` (and their tests) are ported from [nestjs-otel](https://github.com/pragmaticivan/nestjs-otel) (Apache-2.0), with a local `asLangfuseTrace` option added.

## License

Hoard is licensed under [Apache-2.0](https://github.com/hoard-ai/hoard/blob/main/LICENSE).
