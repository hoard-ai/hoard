import type { Uuid } from '@/common/schemas';

import type { CommittedCorefBinding } from '../extraction/types';
import {
  buildCoreferenceBlock,
  buildUnresolvedReferencesBlock,
  type ScopedCandidate,
  selectCandidatesUpToChunk,
  selectCoreferencesForChunks,
} from './coref-utils';
import { selectChunkText } from './text-utils';

const uuid = (n: number): Uuid =>
  `00000000-0000-0000-0000-${String(n).padStart(12, '0')}` as Uuid;

const binding = (
  surfaceForm: string,
  boundNodeId: Uuid,
  sourceChunkIndex: number,
  locatingQuote = `phrase around ${surfaceForm}`,
): CommittedCorefBinding => ({
  surfaceForm,
  boundNodeId,
  sourceChunkIndex,
  locatingQuote,
  resolvedUnresolvedReferenceId: null,
});

describe('coref-utils', () => {
  const nodeNameById = new Map<Uuid, string>([
    [uuid(1), 'Alice'],
    [uuid(2), 'Bob'],
    [uuid(3), 'Antares'],
  ]);

  describe('buildCoreferenceBlock', () => {
    it('returns null when there is nothing to render', () => {
      expect(buildCoreferenceBlock({})).toBeNull();
      expect(buildCoreferenceBlock({ candidates: [], committed: [] })).toBeNull();
    });

    it('renders the entity name, identifyingDescription, and aliases for candidates', () => {
      const block = buildCoreferenceBlock({
        candidates: [
          {
            name: 'Dr. Amara Osei',
            identifyingDescription: 'lead migraine researcher',
            aliases: ['Dr. Osei'],
            referredToAsPronouns: [],
          },
        ],
      });
      expect(block).not.toBeNull();
      expect(block).toContain('Dr. Amara Osei');
      expect(block).toContain('lead migraine researcher');
      expect(block).toContain('Dr. Osei');
    });

    it('renders observed pronouns per candidate and the caveat only when some candidate has them', () => {
      const withPronouns = buildCoreferenceBlock({
        candidates: [
          {
            name: 'Dr. Amara Osei',
            identifyingDescription: 'lead migraine researcher',
            aliases: [],
            referredToAsPronouns: ['she'],
          },
          {
            name: 'Meridian',
            identifyingDescription: 'research vessel',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
      });
      expect(withPronouns).toContain('"she"');
      expect(withPronouns).toContain('pronoun agreement alone never establishes');

      const withoutPronouns = buildCoreferenceBlock({
        candidates: [
          {
            name: 'Meridian',
            identifyingDescription: 'research vessel',
            aliases: [],
            referredToAsPronouns: [],
          },
        ],
      });
      expect(withoutPronouns).not.toContain('refers to them as');
      expect(withoutPronouns).not.toContain('pronoun agreement alone never establishes');
    });

    it('renders committed entries with their locatingQuote quotes', () => {
      const block = buildCoreferenceBlock({
        committed: [
          {
            chunk: 0,
            entries: [
              { surfaceForm: 'she', name: 'Alice', locatingQuote: 'she left early' },
            ],
          },
        ],
      });
      expect(block).toContain('she');
      expect(block).toContain('Alice');
      expect(block).toContain('she left early');
    });

    it('labels groups with the same 1-based [Chunk k] headings as selectChunkText', () => {
      const chunks = ['first chunk', 'second chunk', 'third chunk'];
      const episodeText = selectChunkText(new Set([2]), chunks);

      const block = buildCoreferenceBlock({
        committed: [
          {
            chunk: 2,
            entries: [{ surfaceForm: 'she', name: 'Alice', locatingQuote: 'x' }],
          },
        ],
        labelChunks: true,
      });

      const heading = '[Chunk 3]';
      expect(episodeText).toContain(heading);
      expect(block).toContain(heading);
    });

    it('renders flat without chunk headings when labelChunks is not set', () => {
      const block = buildCoreferenceBlock({
        committed: [
          {
            chunk: 0,
            entries: [{ surfaceForm: 'she', name: 'Alice', locatingQuote: 'x' }],
          },
        ],
      });
      expect(block).not.toContain('[Chunk');
    });
  });

  describe('buildUnresolvedReferencesBlock', () => {
    it('returns null when there are no references', () => {
      expect(buildUnresolvedReferencesBlock([])).toBeNull();
    });

    it('lists references under 0-based refIdx matching array order', () => {
      const block = buildUnresolvedReferencesBlock([
        { surfaceForm: 'she', locatingQuote: 'she found a second colony' },
        { surfaceForm: 'the manager', locatingQuote: 'the manager signed off' },
      ]);
      expect(block).toContain('refIdx 0: "she"');
      expect(block).toContain('she found a second colony');
      expect(block).toContain('refIdx 1: "the manager"');
      expect(block).toContain('unresolvedReferenceIdx');
    });
  });

  describe('selectCandidatesUpToChunk (prefix rule)', () => {
    const candidates: ScopedCandidate[] = [
      {
        name: 'A',
        identifyingDescription: 'a',
        aliases: [],
        referredToAsPronouns: [],
        introChunk: 0,
      },
      {
        name: 'B',
        identifyingDescription: 'b',
        aliases: [],
        referredToAsPronouns: [],
        introChunk: 2,
      },
      {
        name: 'C',
        identifyingDescription: 'c',
        aliases: [],
        referredToAsPronouns: [],
        introChunk: 3,
      },
    ];

    it('keeps only candidates introduced at or before maxChunk', () => {
      const names = selectCandidatesUpToChunk(candidates, 2).map((c) => c.name);
      expect(names).toEqual(['A', 'B']);
    });

    it('drops the scoping field from the result', () => {
      const [first] = selectCandidatesUpToChunk(candidates, 0);
      expect(first).not.toHaveProperty('introChunk');
    });
  });

  describe('selectCoreferencesForChunks (in-view rule)', () => {
    it('keeps only bindings from in-view chunks, grouped and sorted by chunk', () => {
      const bindings = [
        binding('he', uuid(2), 3),
        binding('she', uuid(1), 1),
        binding('her', uuid(1), 2),
      ];
      const groups = selectCoreferencesForChunks(bindings, nodeNameById, new Set([1, 3]));
      expect(groups.map((g) => g.chunk)).toEqual([1, 3]);
      expect(groups[0].entries).toEqual([
        { surfaceForm: 'she', name: 'Alice', locatingQuote: 'phrase around she' },
      ]);
      expect(groups[1].entries.map((e) => e.name)).toEqual(['Bob']);
    });

    it('dedups the same (form, entity) pair within a chunk, keeping the first locatingQuote', () => {
      const bindings = [
        binding('She', uuid(1), 0, 'first occurrence'),
        binding('she', uuid(1), 0, 'second occurrence'),
      ];
      const groups = selectCoreferencesForChunks(bindings, nodeNameById, new Set([0]));
      expect(groups).toHaveLength(1);
      expect(groups[0].entries).toHaveLength(1);
      expect(groups[0].entries[0].locatingQuote).toBe('first occurrence');
    });

    it('keeps the same form bound to different entities within a chunk', () => {
      const bindings = [binding('she', uuid(1), 0), binding('She', uuid(2), 0)];
      const groups = selectCoreferencesForChunks(bindings, nodeNameById, new Set([0]));
      expect(groups[0].entries.map((e) => e.name)).toEqual(['Alice', 'Bob']);
    });

    it('does not dedup the same (form, entity) pair across chunks', () => {
      const bindings = [binding('she', uuid(1), 0), binding('she', uuid(1), 1)];
      const groups = selectCoreferencesForChunks(bindings, nodeNameById, new Set([0, 1]));
      expect(groups).toHaveLength(2);
    });

    it('drops tautological bindings, including a leading article', () => {
      const bindings = [
        binding('the Antares', uuid(3), 0),
        binding('Alice', uuid(1), 0),
        binding('the cutter', uuid(3), 0),
      ];
      const groups = selectCoreferencesForChunks(bindings, nodeNameById, new Set([0]));
      expect(groups).toHaveLength(1);
      expect(groups[0].entries.map((e) => e.surfaceForm)).toEqual(['the cutter']);
    });

    it('throws when an in-view bound entity is missing from the name map', () => {
      const orphan = [binding('they', uuid(9), 1)];
      expect(() =>
        selectCoreferencesForChunks(orphan, nodeNameById, new Set([1])),
      ).toThrow();
    });

    it('ignores out-of-view bindings entirely, including orphans', () => {
      const orphan = [binding('they', uuid(9), 4)];
      expect(selectCoreferencesForChunks(orphan, nodeNameById, new Set([1]))).toEqual([]);
    });
  });
});
