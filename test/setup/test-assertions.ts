import { handler, spec } from 'pactum';
import type { ExpectHandlerContext } from 'pactum/src/exports/handler';

import type { LlmConfigResponse, UserWithoutPassword } from '@test/types';
import {
  LlmConfigResponseSchema,
  LlmProvidersListSchema,
  LoginResponseSchema,
  LogoutResponseSchema,
  SignupResponseSchema,
  TestConfigResponseSchema,
  UserWithoutPasswordSchema,
} from '@test/types';

import { TestHelpers } from './test-helpers';
import { TestSetup } from './test-setup';

export class TestAssertions {
  /**
   * Initialize custom expect handlers
   */
  static initializeHandlers(): void {
    handler.addExpectHandler('validUserResponse', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200 && ctx.res.statusCode !== 201) {
        throw new Error(`Expected status 200 or 201 but got ${ctx.res.statusCode}`);
      }

      const result = UserWithoutPasswordSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid UserWithoutPassword: ${result.error.message}`,
        );
      }
      const body = result.data;

      // Validate specific user data if provided
      const userData = ctx.data as Partial<UserWithoutPassword>;

      if (userData) {
        for (const [key, value] of Object.entries(userData)) {
          if (body[key as keyof UserWithoutPassword] !== value) {
            throw new Error(
              `Expected ${key} to be ${JSON.stringify(value)} but got ${JSON.stringify(body[key as keyof UserWithoutPassword])}`,
            );
          }
        }
      }
    });

    handler.addExpectHandler('successfulSignup', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 201) {
        throw new Error(`Expected status 201 but got ${ctx.res.statusCode}`);
      }

      const result = SignupResponseSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid SignupResponse: ${result.error.message}`,
        );
      }
      const body = result.data;

      if (body.message !== 'Registration successful') {
        throw new Error(
          `Expected message 'Registration successful' but got '${body.message}'`,
        );
      }

      // Validate user data if provided
      const userData = ctx.data as Partial<UserWithoutPassword>;

      if (userData) {
        for (const [key, value] of Object.entries(userData)) {
          if (body.user[key as keyof UserWithoutPassword] !== value) {
            throw new Error(
              `Expected user.${key} to be ${JSON.stringify(value)} but got ${JSON.stringify(body.user[key as keyof UserWithoutPassword])}`,
            );
          }
        }
      }
    });

    handler.addExpectHandler('successfulAuth', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200) {
        throw new Error(`Expected status 200 but got ${ctx.res.statusCode}`);
      }

      const result = LoginResponseSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(`Response is not a valid LoginResponse: ${result.error.message}`);
      }
      const body = result.data;

      if (body.message !== 'Login successful') {
        throw new Error(`Expected message 'Login successful' but got '${body.message}'`);
      }

      // Validate user data if provided
      const userData = ctx.data as Partial<UserWithoutPassword>;

      if (userData) {
        for (const [key, value] of Object.entries(userData)) {
          if (body.user[key as keyof UserWithoutPassword] !== value) {
            throw new Error(
              `Expected user.${key} to be ${JSON.stringify(value)} but got ${JSON.stringify(body.user[key as keyof UserWithoutPassword])}`,
            );
          }
        }
      }
    });

    // Handler for validation errors based on Zod error structure
    handler.addExpectHandler('validationError', (ctx: ExpectHandlerContext) => {
      if (typeof ctx.data !== 'string') {
        throw new Error('Field path must be a string');
      }
      const fieldPath = ctx.data;
      if (!fieldPath) {
        throw new Error('Field path is required for validation error handler');
      }

      if (ctx.res.statusCode !== 400) {
        throw new Error(`Expected status 400 but got ${ctx.res.statusCode}`);
      }

      const body = ctx.res.json as {
        statusCode: number;
        message: string;
        errors: Array<{ code: string; path: string[]; message: string }>;
      };

      if (
        typeof body.statusCode !== 'number' ||
        typeof body.message !== 'string' ||
        !Array.isArray(body.errors)
      ) {
        throw new Error('Response is not a valid validation error response');
      }

      const hasFieldError = body.errors.some(
        (error) =>
          (error.path && error.path.includes(fieldPath)) ||
          (error.code === 'custom' && fieldPath === 'custom'),
      );
      if (!hasFieldError) {
        throw new Error(
          `Expected validation error for field '${fieldPath}' but not found in errors`,
        );
      }
    });

    handler.addExpectHandler('authFailure', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 403) {
        throw new Error(`Expected status 403 but got ${ctx.res.statusCode}`);
      }
    });

    handler.addExpectHandler('validLlmConfigResponse', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200) {
        throw new Error(`Expected status 200 but got ${ctx.res.statusCode}`);
      }

      const result = LlmConfigResponseSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid LlmConfigResponse: ${result.error.message}`,
        );
      }
      const body = result.data as Record<string, unknown>;
      const assertions = ctx.data as Partial<LlmConfigResponse> | undefined;

      if (assertions) {
        for (const [key, value] of Object.entries(assertions)) {
          if (body[key] !== value) {
            throw new Error(
              `Expected ${key} to be ${String(value)} but got ${String(body[key])}`,
            );
          }
        }
      }
    });

    handler.addExpectHandler('validProvidersList', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200) {
        throw new Error(`Expected status 200 but got ${ctx.res.statusCode}`);
      }

      const result = LlmProvidersListSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid LlmProvidersList: ${result.error.message}`,
        );
      }
      const body = result.data;

      const assertions = ctx.data as { activeProvider?: string | null } | undefined;

      if (assertions?.activeProvider !== undefined) {
        if (body.activeProvider !== assertions.activeProvider) {
          throw new Error(
            `Expected activeProvider to be ${String(assertions.activeProvider)} but got ${String(body.activeProvider)}`,
          );
        }
      }
    });

    handler.addExpectHandler('validTestConfigResponse', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200) {
        throw new Error(`Expected status 200 but got ${ctx.res.statusCode}`);
      }

      const result = TestConfigResponseSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid TestConfigResponse: ${result.error.message}`,
        );
      }
      const body = result.data;
      const assertions = ctx.data as { success?: boolean } | undefined;

      if (assertions?.success !== undefined && body.success !== assertions.success) {
        throw new Error(
          `Expected success to be ${String(assertions.success)} but got ${String(body.success)}`,
        );
      }
    });

    handler.addExpectHandler('successfulLogout', (ctx: ExpectHandlerContext) => {
      if (ctx.res.statusCode !== 200) {
        throw new Error(`Expected status 200 but got ${ctx.res.statusCode}`);
      }

      // Use Zod to validate the response
      const result = LogoutResponseSchema.safeParse(ctx.res.json);
      if (!result.success) {
        throw new Error(
          `Response is not a valid LogoutResponse: ${result.error.message}`,
        );
      }

      const body = result.data;
      if (body.message !== 'Logout successful') {
        throw new Error(`Expected message 'Logout successful' but got '${body.message}'`);
      }
    });
  }

  /**
   * Validate authenticated access to protected endpoint
   */
  static async validateAuthenticatedAccess(
    sessionKey: string,
  ): Promise<UserWithoutPassword> {
    return await TestHelpers.authenticatedRequest(sessionKey)
      .get(`${TestSetup.baseUrl}/users/me`)
      .expect('validUserResponse')
      .returns('');
  }

  /**
   * Validate unauthenticated rejection for endpoints
   */
  static async validateUnauthenticatedRejection(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET',
  ): Promise<void> {
    const url = `${TestSetup.baseUrl}${endpoint}`;

    switch (method) {
      case 'GET':
        await spec().get(url).expect('authFailure');
        break;
      case 'POST':
        await spec().post(url).expect('authFailure');
        break;
      case 'PUT':
        await spec().put(url).expect('authFailure');
        break;
      case 'DELETE':
        await spec().delete(url).expect('authFailure');
        break;
      default:
        throw new Error(`Unsupported HTTP method: ${method as string}`);
    }
  }
}
