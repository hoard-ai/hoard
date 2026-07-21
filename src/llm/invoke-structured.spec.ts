import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { mockDeep } from 'jest-mock-extended';
import { z } from 'zod';

import { invokeStructured, StructuredOutputValidationError } from './invoke-structured';

const EdgeSchema = z.object({
  fact: z.string(),
  validAt: z.iso.datetime().nullable().optional(),
});

const baseMessages = [new SystemMessage('extract'), new HumanMessage('source text')];

describe('invokeStructured', () => {
  let mockModel: ReturnType<typeof mockDeep<BaseChatModel>>;
  let mockRunnable: { invoke: jest.Mock };

  beforeEach(() => {
    mockModel = mockDeep<BaseChatModel>();
    mockRunnable = { invoke: jest.fn() };
    mockModel.withStructuredOutput.mockReturnValue(mockRunnable as never);
  });

  afterEach(() => jest.clearAllMocks());

  it('returns parsed value on first-attempt success', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'Alice works at Acme',
      validAt: '2025-04-30T00:00:00Z',
    });

    const result = await invokeStructured(mockModel, EdgeSchema, baseMessages, {
      runName: 'extract-test',
      tags: ['test'],
    });

    expect(result).toEqual({
      fact: 'Alice works at Acme',
      validAt: '2025-04-30T00:00:00Z',
    });
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(1);
    expect(mockRunnable.invoke).toHaveBeenCalledWith(
      baseMessages,
      expect.objectContaining({ runName: 'extract-test', tags: ['test'] }),
    );
  });

  it('retries with appended HumanMessage on ZodError, then succeeds', async () => {
    mockRunnable.invoke
      .mockResolvedValueOnce({ fact: 'Alice works at Acme', validAt: 'next tuesday' })
      .mockResolvedValueOnce({
        fact: 'Alice works at Acme',
        validAt: '2025-04-30T00:00:00Z',
      });

    const result = await invokeStructured(mockModel, EdgeSchema, baseMessages, {
      runName: 'extract-test',
      tags: ['test'],
    });

    expect(result.validAt).toBe('2025-04-30T00:00:00Z');
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(2);

    const [secondMessages, secondConfig] = mockRunnable.invoke.mock.calls[1] as [
      Array<AIMessage | HumanMessage | SystemMessage>,
      { runName?: string; tags?: string[] },
    ];
    // Retry echoes the rejected output (AIMessage) then the feedback (HumanMessage).
    expect(secondMessages).toHaveLength(baseMessages.length + 2);
    const echoed = secondMessages[secondMessages.length - 2];
    expect(echoed).toBeInstanceOf(AIMessage);
    expect(echoed.content as string).toContain('next tuesday');
    expect(secondMessages[secondMessages.length - 1]).toBeInstanceOf(HumanMessage);
    expect(secondConfig.runName).toBe('extract-test.retry-1');
    expect(secondConfig.tags).toEqual(['test', 'retry']);
  });

  it('throws StructuredOutputValidationError after exhausting attempts', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'Alice works at Acme',
      validAt: 'never',
    });

    let caught: unknown;
    try {
      await invokeStructured(mockModel, EdgeSchema, baseMessages, {
        runName: 'extract-test',
        maxRetries: 2,
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(StructuredOutputValidationError);
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(2);

    const err = caught as StructuredOutputValidationError;
    expect(err.runName).toBe('extract-test');
    expect(err.issues.length).toBeGreaterThan(0);
    expect(err.message).not.toContain('Alice works at Acme');
    // issues are structural `path: zod-code`, never the raw value `never`.
    expect(err.issues.every((s) => !s.includes('never'))).toBe(true);
    expect(err.issues.some((s) => s.startsWith('validAt:'))).toBe(true);
  });

  it('does not retry when maxRetries=1', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'Alice works at Acme',
      validAt: 'never',
    });

    await expect(
      invokeStructured(mockModel, EdgeSchema, baseMessages, {
        runName: 'extract-test',
        maxRetries: 1,
      }),
    ).rejects.toBeInstanceOf(StructuredOutputValidationError);

    expect(mockRunnable.invoke).toHaveBeenCalledTimes(1);
  });

  it('retries with appended HumanMessage on invariant violation, then succeeds', async () => {
    mockRunnable.invoke
      .mockResolvedValueOnce({ fact: 'wrong', validAt: '2025-04-30T00:00:00Z' })
      .mockResolvedValueOnce({ fact: 'right', validAt: '2025-04-30T00:00:00Z' });

    const validate = jest
      .fn()
      .mockReturnValueOnce([{ code: 'fact.wrong', message: 'fact must be "right"' }])
      .mockReturnValueOnce([]);

    const result = await invokeStructured(mockModel, EdgeSchema, baseMessages, {
      runName: 'extract-test',
      validate,
    });

    expect(result.fact).toBe('right');
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(2);
    expect(validate).toHaveBeenCalledTimes(2);

    const [secondMessages] = mockRunnable.invoke.mock.calls[1] as [
      Array<AIMessage | HumanMessage | SystemMessage>,
    ];
    expect(secondMessages).toHaveLength(baseMessages.length + 2);
    expect(secondMessages[secondMessages.length - 2]).toBeInstanceOf(AIMessage);
    expect(secondMessages[secondMessages.length - 1]).toBeInstanceOf(HumanMessage);
  });

  it('throws StructuredOutputValidationError after exhausting attempts on persistent violations', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'wrong',
      validAt: '2025-04-30T00:00:00Z',
    });
    const validate = jest
      .fn()
      .mockReturnValue([{ code: 'fact.wrong', message: 'fact must be "right"' }]);

    let caught: unknown;
    try {
      await invokeStructured(mockModel, EdgeSchema, baseMessages, {
        runName: 'extract-test',
        maxRetries: 2,
        validate,
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(StructuredOutputValidationError);
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(2);
    const err = caught as StructuredOutputValidationError;
    expect(err.runName).toBe('extract-test');
    expect(err.issues.length).toBeGreaterThan(0);
  });

  it('feeds verbose message to LLM retry but keeps structural code in error', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'wrong',
      validAt: '2025-04-30T00:00:00Z',
    });
    const validate = jest.fn().mockReturnValue([
      {
        code: 'fact.unknown-entity',
        message: 'name "Alice Smith" is not in the input ENTITIES set',
      },
    ]);

    let caught: unknown;
    try {
      await invokeStructured(mockModel, EdgeSchema, baseMessages, {
        runName: 'extract-test',
        maxRetries: 2,
        validate,
      });
    } catch (e) {
      caught = e;
    }

    const [retryMessages] = mockRunnable.invoke.mock.calls[1] as [
      Array<HumanMessage | SystemMessage>,
    ];
    const retryFeedback = retryMessages[retryMessages.length - 1] as HumanMessage;
    const retryContent = retryFeedback.content as string;
    expect(retryContent).toContain('Alice Smith');

    const err = caught as StructuredOutputValidationError;
    expect(err.issues).toEqual(['fact.unknown-entity']);
    expect(err.message).not.toContain('Alice Smith');
  });

  it('skips validate when not provided', async () => {
    mockRunnable.invoke.mockResolvedValue({
      fact: 'anything',
      validAt: '2025-04-30T00:00:00Z',
    });

    const result = await invokeStructured(mockModel, EdgeSchema, baseMessages, {
      runName: 'extract-test',
    });

    expect(result.fact).toBe('anything');
    expect(mockRunnable.invoke).toHaveBeenCalledTimes(1);
  });

  it('passes callbacks through to invoke', async () => {
    mockRunnable.invoke.mockResolvedValue({ fact: 'x' });
    const callbacks = [{ name: 'fake' }] as never;

    await invokeStructured(mockModel, EdgeSchema, baseMessages, {
      runName: 'extract-test',
      callbacks,
    });

    expect(mockRunnable.invoke).toHaveBeenCalledWith(
      baseMessages,
      expect.objectContaining({ callbacks }),
    );
  });
});
