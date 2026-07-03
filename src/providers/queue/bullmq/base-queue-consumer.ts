import { WorkerHost } from '@nestjs/bullmq';
import { Logger, OnApplicationBootstrap, OnModuleDestroy } from '@nestjs/common';
import { Queue } from 'bullmq';

import { Environment } from '@/config/app/configuration';

/**
 * Abstract base class for BullMQ queue consumers with test-aware cleanup.
 * Provides:
 * - Automatic logger initialization
 * - Graceful shutdown with worker cleanup
 * - Test teardown error detection method to avoid logging noise
 * - Lock-loss cancellation: aborts a job's processor signal when its lock
 *   renewal fails, so consumers that accept the signal can unwind in-flight work
 */
export abstract class BaseQueueConsumer
  extends WorkerHost
  implements OnApplicationBootstrap, OnModuleDestroy
{
  protected readonly logger: Logger;
  protected readonly isTest = process.env.NODE_ENV === Environment.Test;

  constructor(
    loggerContext: string,
    protected readonly queue: Queue,
  ) {
    super();
    this.logger = new Logger(loggerContext);
  }

  /**
   * Wire lock-loss cancellation once the worker exists (set during the BullMQ
   * explorer's module init). When lock renewal fails we abort the job's
   * processor signal via cancelJob().
   *
   * Important: a worker that has lost a job's lock cannot move that job to
   * `failed` (it no longer owns the lock). Instead cancelJob() aborts the
   * signal so the processor can clean up; the job stays `active` briefly until
   * BullMQ's stalled-job checker moves it back to `waiting` and another worker
   * retries it. This is the intended behavior - trust the stalled-job mechanism
   * rather than trying to fail the job here.
   * Docs: https://docs.bullmq.io/guide/workers/cancelling-jobs (accessed 2026-07-02)
   * https://web.archive.org/web/20260702200313/https://docs.bullmq.io/guide/workers/cancelling-jobs
   */
  onApplicationBootstrap(): void {
    this.worker.on('lockRenewalFailed', (jobIds: string[]) => {
      this.logger.error(
        `Lock renewal failed on ${this.queue.name} for ${jobIds.length} job(s): ${jobIds.join(', ')}`,
      );
      for (const jobId of jobIds) {
        this.worker.cancelJob(jobId, 'lock renewal failed');
      }
    });
  }

  /**
   * Check if error is a pool closure error during test teardown.
   * Use this to avoid logging expected test teardown errors.
   */
  protected isTestTeardownError(error: unknown): boolean {
    return (
      this.isTest &&
      error instanceof Error &&
      error.message?.includes('Cannot use a pool after calling end')
    );
  }

  async onModuleDestroy() {
    if (!this.worker) return;

    await this.worker.close(this.isTest);

    if (!this.queue.closing) {
      await this.queue.close();
    }
  }
}
