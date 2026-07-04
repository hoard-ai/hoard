import type { INestApplication } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import { AppModule } from '../../src/app.module';
import { DatabaseUtils } from './database-utils';

export class TestSetup {
  static app: INestApplication;
  static module: TestingModule;
  static baseUrl = `http://localhost:${process.env.PORT ?? 3333}`;

  static async setupApp(): Promise<void> {
    this.module = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    this.app = this.module.createNestApplication();
    await this.app.init();
    await this.app.listen(process.env.PORT ?? 3333);

    await DatabaseUtils.cleanDatabase();
  }

  static async teardownApp(): Promise<void> {
    await DatabaseUtils.cleanDatabase();
    await DatabaseUtils.cleanup();
    await this.app.close();

    // Close the module to properly cleanup BullMQ and other connections
    // For some reason issue only appeared after Prisma v7 migration
    // https://stackoverflow.com/questions/62975121/close-redis-connection-when-using-nestjs-queues
    // https://stackoverflow.com/a/66246647
    // Page saved in WaybackMachine
    await this.module.close();
  }
}
