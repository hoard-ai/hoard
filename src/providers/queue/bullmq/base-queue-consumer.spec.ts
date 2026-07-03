import { Logger } from '@nestjs/common';
import { Job, Queue } from 'bullmq';

import { BaseQueueConsumer } from './base-queue-consumer';

class TestConsumer extends BaseQueueConsumer {
  constructor(queue: Queue) {
    super('TestConsumer', queue);
  }
  async process(_job: Job): Promise<void> {}
}

describe('BaseQueueConsumer', () => {
  function setup() {
    const handlers: Record<string, (arg: unknown) => void> = {};
    const worker = {
      on: jest.fn((event: string, handler: (arg: unknown) => void) => {
        handlers[event] = handler;
      }),
      cancelJob: jest.fn(),
    };
    const queue = { name: 'test-queue' } as unknown as Queue;
    const consumer = new TestConsumer(queue);
    // Mirror what the BullMQ explorer does at bootstrap: attach the worker.
    (consumer as unknown as { _worker: unknown })._worker = worker;
    const errorSpy = jest.spyOn(Logger.prototype, 'error').mockImplementation(() => {});
    return { consumer, worker, handlers, errorSpy };
  }

  afterEach(() => jest.restoreAllMocks());

  it('registers a lockRenewalFailed handler on bootstrap', () => {
    const { consumer, worker } = setup();
    consumer.onApplicationBootstrap();
    expect(worker.on).toHaveBeenCalledWith('lockRenewalFailed', expect.any(Function));
  });

  it('cancels every job whose lock renewal failed', () => {
    const { consumer, worker, handlers, errorSpy } = setup();
    consumer.onApplicationBootstrap();

    handlers['lockRenewalFailed'](['j1', 'j2']);

    expect(worker.cancelJob).toHaveBeenCalledTimes(2);
    expect(worker.cancelJob).toHaveBeenCalledWith('j1', 'lock renewal failed');
    expect(worker.cancelJob).toHaveBeenCalledWith('j2', 'lock renewal failed');
    expect(errorSpy).toHaveBeenCalledTimes(1);
  });
});
