import { spec } from 'pactum';

import type { User } from '@test/types';
import { Role } from '@test/types';

import { DatabaseUtils } from './database-utils';
import { TestDataFactory } from './test-data-factory';
import { TestSetup } from './test-setup';

export interface SessionResult {
  userId: string;
  credentials: {
    email: string;
    password: string;
  };
  sessionKey: string;
  userData: User;
}

export class TestHelpers {
  private static sessionCounter = 0;
  private static prisma = DatabaseUtils.getPrismaClient();

  static generateSessionKey(prefix = 'session'): string {
    return `${prefix}_${++this.sessionCounter}_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Create authenticated spec with session cookie
   */
  static authenticatedRequest(sessionKey: string) {
    return spec().withCookies(`$S{${sessionKey}}`);
  }

  /**
   * Create user only (no login)
   */
  static async createUser(
    userPrefix = 'test',
  ): Promise<{ id: string; credentials: { email: string; password: string } }> {
    const credentials = TestDataFactory.createUserCredentials(userPrefix);
    const id: string = await spec()
      .post(`${TestSetup.baseUrl}/auth/signup`)
      .withBody(credentials)
      .expectStatus(201)
      .returns('user.id');

    return { id, credentials };
  }

  /**
   * Login existing user and return session info
   */
  static async loginUser(userCredentials: {
    email: string;
    password: string;
  }): Promise<SessionResult> {
    const sessionKey = this.generateSessionKey();
    const userData: User = await spec()
      .post(`${TestSetup.baseUrl}/auth/login`)
      .withBody(userCredentials)
      .expectStatus(200)
      .stores(sessionKey, 'res.headers.set-cookie')
      .returns('user');

    return {
      userId: userData.id,
      credentials: userCredentials,
      sessionKey,
      userData,
    };
  }

  /**
   * Logout user by sessionKey
   */
  static async logoutUser(sessionKey: string): Promise<void> {
    await this.authenticatedRequest(sessionKey)
      .post(`${TestSetup.baseUrl}/auth/logout`)
      .expectStatus(200);
  }

  /**
   * Create and login a user in one step
   */
  static async createUserWithSession(userPrefix = 'test'): Promise<SessionResult> {
    const { credentials } = await this.createUser(userPrefix);

    return await this.loginUser(credentials);
  }

  /**
   * Create admin only (no login)
   */
  static async createAdmin(
    adminPrefix = 'admin',
  ): Promise<{ id: string; credentials: { email: string; password: string } }> {
    const { id, credentials } = await this.createUser(adminPrefix);

    await this.promoteUserToAdmin(id);

    return { id, credentials };
  }

  /**
   * Create and login an admin in one step
   */
  static async createAdminWithSession(adminPrefix = 'admin'): Promise<SessionResult> {
    const { id: userId, credentials: adminCredentials } =
      await this.createUser(adminPrefix);

    await this.promoteUserToAdmin(userId);

    return await this.loginUser(adminCredentials);
  }

  /**
   * Promote an existing user to admin role
   */
  static async promoteUserToAdmin(userId: string): Promise<void> {
    await this.prisma.user.update({
      where: { id: userId },
      data: { role: Role.ADMIN },
    });
  }
}
