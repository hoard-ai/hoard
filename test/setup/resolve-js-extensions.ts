/**
 * Jest globalSetup/globalTeardown load their dependency graph through native Node
 * resolution (patched only by tsconfig-paths, for aliases), so jest's
 * moduleNameMapper never sees it. Prisma v7 under `nodenext` emits relative
 * imports with `.js` specifiers (e.g. "./internal/class.js") that point at `.ts`
 * sources. The `.ts` extension is already registered for these entry points, so
 * strip `.js` from relative specifiers to let native resolution find the source.
 */
import Module from 'node:module';

type ResolveFilename = (this: unknown, request: string, ...rest: unknown[]) => string;

const patched = Module as unknown as { _resolveFilename: ResolveFilename };
const originalResolveFilename = patched._resolveFilename;

patched._resolveFilename = function (request, ...rest) {
  if (/^\.{1,2}\/.*\.js$/.test(request)) {
    try {
      return originalResolveFilename.call(this, request.replace(/\.js$/, ''), ...rest);
    } catch {
      return originalResolveFilename.call(this, request, ...rest);
    }
  }
  return originalResolveFilename.call(this, request, ...rest);
};
