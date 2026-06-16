# Hoard API

Hoard is an automatically managed, graph-enabled, time-aware knowledge bank:
a provider-agnostic memory layer that any AI agent can plug into. RememberNow
is one consumer of Hoard.

The point is to allow users (or agents) to dump any amount of information into
Hoard, and it will embed it into the graph, reminding people of things when
necessary (e.g. appointments).

This is a GraphRAG application, the core of which is in `src/knowledge-graph`.

It is a modified and improved TypeScript port of Graphiti by Zep, which is written in Python originally.

The original Graphiti codebase can only be accessed if the
user cloned the repository. Check existence with `ls graphiti/`
if you deem referencing necessary.

Tech stack:
NestJS 11
Session auth
TypeScript (strict, no `any`)
LangChain + LangGraph
Zod validation
Swagger docs generated automatically using `nestjs-zod`, as all DTOs (in, out) are specified
BullMQ (Redis)
Prisma (PostgreSQL + pgvector + pgvectorscale for ANN search)

All environment variables are split by domain and loaded in `src/config`.

Everything is eventually unit tested, and e2e tested. Unit tests are in `src/`, and they use helpers (e.g. factories) from `src/test`.

E2e tests are in `test/`.

Important: only run unit tests with `--runInBand` set as app compilation cost
is heavy.

There are pre-commit hooks - `npm run lint` and `npm run format`.

When implementing things, please use context7 as it allows you to fetch all
relevant documentation, and don't forget to lint and format at the end.

Prefer to use `npm run format:diff` instead of `npm run format` as
it only formats files that were changed in git, and is significantly faster.

When `format`/`lint --fix` reports it reformatted files after the build and
tests already passed, do NOT re-run the build or tests. Those tools only make
cosmetic changes (whitespace, import order, line wrapping) that cannot break a
green build - re-running is wasted time.

Never add "migration code" to make sure the code is compatible with previous
versions of the graph. If something requires a manual migration, tell that
to me explicitly.

Never pipe `npm run lint` into something like `tail` or `grep` because the lint
takes 22 seconds to run, so you're wasting time. The output isn't that big.
Same for `npm test`.

Use `npm run prisma:generate` instead of `npx prisma generate `since it automatically uses dotenv to load the proper .env file when calling prisma. Same for other prisma commands.

## Conventions

- In the knowledge-graph module, DB access lives only in `src/knowledge-graph/repository/repositories`; its services/helpers consume repo methods, never Prisma directly. (This is a KG-module rule, not a top-level API rule.)
- Module types go in `src/<module>/types/<module>.types.ts` with a `types/index.ts` barrel.
- Never swallow exceptions — throw or log.
- A value that should always be present but is typed optional (`?:` / `| undefined`) is debt — make required things required, and throw on a missing value rather than silently defaulting when the type is shared with a path that legitimately lacks it.
- No `eslint-disable` for typing rules — restructure types (`unknown` over `any`, drop needless `async`).
- Don't re-`parse()` in repo methods — callers already validated.
- Errors/logs on LLM paths must not include raw model output (PII via entity names/facts) — surface only structural info / ZodError issues.
- Avoid PII in logs / traces at all costs.
- Comments: inline comments inside a body are one short line max, no multi-line explanatory blocks. A function/method/class's own description belongs in a `/** */` JSDoc docstring above the declaration, not `//` line comments (the one-line rule is for inline body comments, not the doc header).
- API stays presentation-agnostic — no colors, sizes, coordinates, or layout; the frontend owns visuals.
- Omit `LlmContext.sessionId` unless there's a real session/conversation; never fall back to userId/graphId/jobId.
- Prompt text: ASCII hyphens (no em dashes), real-newline wrapping (no trailing `\`), schema field refs in camelCase.
- Tests: assert shape (length/instanceof/empty), not exact error strings; derive enum-driven values from Zod schemas instead of hardcoding.
  We test logic, rather than exact wording the developer specified.

Lastly, `tsconfig.json`:

```json
"paths": {
    "@/*": ["./src/*"],
    "@generated/*": ["./generated/*"],
    "@test/*": ["./test/*"]
}
```
