import { spec } from 'pactum';

import {
  TestAssertions as a,
  TestDataFactory as f,
  TestHelpers as h,
  TestSetup as s,
} from '@test/setup';
import type { PaginatedUsers, User } from '@test/types';

describe('User Admin (e2e)', () => {
  describe('POST /users (Admin Only)', () => {
    describe('Success Cases', () => {
      it('should create user successfully as admin', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-create');
        const newUserData = {
          ...f.createUserCredentials('new'),
          role: 'USER',
        };

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody(newUserData)
          .expectStatus(201)
          .expect('validUserResponse', {
            email: newUserData.email,
            role: 'USER',
          });
      });

      it('should create admin user successfully', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-create-admin');
        const newAdminData = {
          ...f.createUserCredentials('new-admin'),
          role: 'ADMIN',
        };

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody(newAdminData)
          .expectStatus(201)
          .expect('validUserResponse', {
            email: newAdminData.email,
            role: 'ADMIN',
          });
      });

      it('should default to USER role when not specified', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-default-role');
        const newUserData = f.createUserCredentials('default-role');

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody(newUserData)
          .expectStatus(201)
          .expect('validUserResponse', {
            role: 'USER',
          });
      });

      it('should trim and lowercase email', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-trim-email');
        const emailWithSpaces = f.EMAIL_PATTERNS.withSpaces('admin-trim');
        const expectedEmail = emailWithSpaces.trim().toLowerCase();

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody({
            email: emailWithSpaces,
            password: f.PASSWORD_PATTERNS.valid(),
          })
          .expectStatus(201)
          .expectBodyContains(expectedEmail);
      });
    });

    describe('Validation Errors', () => {
      it('should fail with duplicate email', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-duplicate');
        const existingUser = await h.createUser('existing');

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody({
            email: existingUser.credentials.email,
            password: f.PASSWORD_PATTERNS.valid(),
          })
          .expect('authFailure');
      });

      it('should fail with various validation scenarios', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-validation');
        const scenarios = f.createSignupValidationScenarios();

        for (const { data, expectedField } of scenarios) {
          await h
            .authenticatedRequest(sessionKey)
            .post(`${s.baseUrl}/users`)
            .withBody(data)
            .expect('validationError', expectedField);
        }
      });

      it('should fail with invalid role', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-invalid-role');

        await h
          .authenticatedRequest(sessionKey)
          .post(`${s.baseUrl}/users`)
          .withBody({
            email: f.EMAIL_PATTERNS.valid('invalid-role'),
            password: f.PASSWORD_PATTERNS.valid(),
            role: 'INVALID_ROLE',
          })
          .expect('validationError', 'role');
      });
    });

    describe('Authentication Failures', () => {
      it('should fail without authentication', async () => {
        await spec()
          .post(`${s.baseUrl}/users`)
          .withBody(f.createUserCredentials('unauthenticated'))
          .expect('authFailure');
      });

      it('should fail with non-admin user', async () => {
        const { sessionKey } = await h.createUserWithSession('regular-user');

        await spec()
          .post(`${s.baseUrl}/users`)
          .withCookies(`$S{${sessionKey}}`)
          .withBody(f.createUserCredentials('unauthorized'))
          .expect('authFailure');
      });
    });
  });

  describe('GET /users (Admin Only)', () => {
    describe('Success Cases', () => {
      it('should return paginated users', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-list');

        await h.createUser('list-1');
        await h.createUser('list-2');
        await h.createUser('list-3');

        const response: PaginatedUsers = await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users`)
          .expectStatus(200)
          .returns('');

        expect(response.users).toBeInstanceOf(Array);
        expect(response.pagination).toBeDefined();
        expect(response.pagination.page).toBe(1);
        expect(response.pagination.limit).toBe(10);
        expect(response.users.length).toBeGreaterThan(0);
      });

      it('should return users with pagination parameters', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-pagination');

        const response: PaginatedUsers = await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users`)
          .withQueryParams('page', '1')
          .withQueryParams('limit', '5')
          .expectStatus(200)
          .returns('');

        expect(response.pagination.page).toBe(1);
        expect(response.pagination.limit).toBe(5);
      });

      it('should search users by email', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-search');
        await h.createUser('searchable');

        const response: PaginatedUsers = await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users`)
          .withQueryParams('search', 'searchable')
          .expectStatus(200)
          .returns('');

        expect(response.users.some((user) => user.email.includes('searchable'))).toBe(
          true,
        );
      });
    });

    describe('Validation Errors', () => {
      it('should fail with invalid page parameter', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-invalid-page');

        await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users`)
          .withQueryParams('page', '0')
          .expect('validationError', 'page');
      });

      it('should fail with invalid limit parameter', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-invalid-limit');

        await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users`)
          .withQueryParams('limit', '101')
          .expect('validationError', 'limit');
      });
    });

    describe('Authentication Failures', () => {
      it('should fail without authentication', async () => {
        await a.validateUnauthenticatedRejection('/users');
      });

      it('should fail with non-admin user', async () => {
        const { sessionKey } = await h.createUserWithSession('regular-user');

        await spec()
          .get(`${s.baseUrl}/users`)
          .withCookies(`$S{${sessionKey}}`)
          .expect('authFailure');
      });
    });
  });

  describe('GET /users/:id (Admin Only)', () => {
    describe('Success Cases', () => {
      it('should return specific user by id', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-get-by-id');
        const targetUser = await h.createUser('target');

        await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users/${targetUser.id}`)
          .expectStatus(200)
          .expectBodyContains(targetUser.credentials.email);
      });

      it('should include password hash for getting user by id', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-get-full');
        const targetUser = await h.createUser('full-access');

        const response: User = await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users/${targetUser.id}`)
          .expectStatus(200)
          .returns('');

        expect(response.passwordHash).toBeDefined();
        expect(typeof response.passwordHash).toBe('string');
      });
    });

    describe('Error Cases', () => {
      it('should fail with non-existent user id', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-not-found');

        await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users/00000000-0000-4000-8000-000000000999`)
          .expectStatus(404);
      });

      it('should fail with invalid user id format', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-invalid-id');

        await h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users/invalid`)
          .expect('validationError', 'id');
      });
    });

    describe('Authentication Failures', () => {
      it('should fail without authentication', async () => {
        await a.validateUnauthenticatedRejection('/users/1');
      });

      it('should fail with non-admin user', async () => {
        const { sessionKey } = await h.createUserWithSession('regular-user');

        await spec()
          .get(`${s.baseUrl}/users/1`)
          .withCookies(`$S{${sessionKey}}`)
          .expect('authFailure');
      });
    });
  });

  describe('PUT /users/:id (Admin Only)', () => {
    describe('Success Cases', () => {
      it('should update user email', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-update-email');
        const targetUser = await h.createUser('update-target');
        const newEmail = f.EMAIL_PATTERNS.valid('updated');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ email: newEmail })
          .expectStatus(200)
          .expectBodyContains(newEmail);
      });

      it('should update user password', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-update-password');
        const targetUser = await h.createUser('password-update');
        const newPassword = f.PASSWORD_PATTERNS.valid();

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ password: newPassword })
          .expectStatus(200);

        // Verify user can login with new password
        await spec()
          .post(`${s.baseUrl}/auth/login`)
          .withBody({
            email: targetUser.credentials.email,
            password: newPassword,
          })
          .expectStatus(200);
      });

      it('should update user role', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-update-role');
        const targetUser = await h.createUser('role-update');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ role: 'ADMIN' })
          .expectStatus(200)
          .expectBodyContains('ADMIN');
      });

      it('should update multiple fields at once', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-update-multiple');
        const targetUser = await h.createUser('multiple-update');
        const newEmail = f.EMAIL_PATTERNS.valid('multi-updated');
        const newPassword = f.PASSWORD_PATTERNS.valid();

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({
            email: newEmail,
            password: newPassword,
            role: 'ADMIN',
          })
          .expectStatus(200)
          .expectBodyContains(newEmail)
          .expectBodyContains('ADMIN');

        // Verify login with new credentials
        await spec()
          .post(`${s.baseUrl}/auth/login`)
          .withBody({
            email: newEmail,
            password: newPassword,
          })
          .expectStatus(200);
      });

      it('should trim and lowercase email updates', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-trim-update');
        const targetUser = await h.createUser('trim-update');
        const emailWithSpaces = f.EMAIL_PATTERNS.withSpaces('trimmed-update');
        const expectedEmail = emailWithSpaces.trim().toLowerCase();

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ email: emailWithSpaces })
          .expectStatus(200)
          .expectBodyContains(expectedEmail);
      });
    });

    describe('Validation Errors', () => {
      it('should fail when no fields provided', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-no-fields');
        const targetUser = await h.createUser('no-fields');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({})
          .expectStatus(400);
      });

      it('should fail with duplicate email', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-duplicate-update');
        const existingUser = await h.createUser('existing-update');
        const targetUser = await h.createUser('target-update');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ email: existingUser.credentials.email })
          .expect('authFailure');
      });

      it('should fail with invalid email format', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-invalid-email');
        const targetUser = await h.createUser('invalid-email-update');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ email: 'invalid-email-format' })
          .expect('validationError', 'email');
      });

      it('should fail with short password', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-short-password');
        const targetUser = await h.createUser('short-password');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ password: '123' })
          .expect('validationError', 'password');
      });

      it('should fail with invalid role', async () => {
        const { sessionKey } = await h.createAdminWithSession(
          'admin-invalid-role-update',
        );
        const targetUser = await h.createUser('invalid-role-update');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withBody({ role: 'INVALID_ROLE' })
          .expect('validationError', 'role');
      });
    });

    describe('Error Cases', () => {
      it('should fail with non-existent user id', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-update-not-found');

        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/00000000-0000-4000-8000-000000000999`)
          .withBody({ email: f.EMAIL_PATTERNS.valid('not-found') })
          .expectStatus(404);
      });
    });

    describe('Authentication Failures', () => {
      it('should fail without authentication', async () => {
        await spec()
          .put(`${s.baseUrl}/users/1`)
          .withBody({ email: f.EMAIL_PATTERNS.valid('unauthenticated') })
          .expect('authFailure');
      });

      it('should fail with non-admin user', async () => {
        const { sessionKey } = await h.createUserWithSession('regular-user');
        const targetUser = await h.createUser('unauthorized-target');

        await spec()
          .put(`${s.baseUrl}/users/${targetUser.id}`)
          .withCookies(`$S{${sessionKey}}`)
          .withBody({ email: f.EMAIL_PATTERNS.valid('unauthorized') })
          .expect('authFailure');
      });
    });
  });

  describe('DELETE /users/:id (Admin Only)', () => {
    describe('Success Cases', () => {
      it('should delete user successfully', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-delete');
        const targetUser = await h.createUser('delete-target');

        await h
          .authenticatedRequest(sessionKey)
          .delete(`${s.baseUrl}/users/${targetUser.id}`)
          .expectStatus(204);

        // Verify user can no longer login
        await spec()
          .post(`${s.baseUrl}/auth/login`)
          .withBody(targetUser.credentials)
          .expect('authFailure');
      });

      it('should delete admin user successfully', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-delete-admin');
        const targetAdmin = await h.createAdmin('delete-admin-target');

        await h
          .authenticatedRequest(sessionKey)
          .delete(`${s.baseUrl}/users/${targetAdmin.id}`)
          .expectStatus(204);

        // Verify admin can no longer login
        await spec()
          .post(`${s.baseUrl}/auth/login`)
          .withBody(targetAdmin.credentials)
          .expect('authFailure');
      });
    });

    describe('Error Cases', () => {
      it('should fail with non-existent user id', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-delete-not-found');

        await h
          .authenticatedRequest(sessionKey)
          .delete(`${s.baseUrl}/users/00000000-0000-4000-8000-000000000999`)
          .expectStatus(404);
      });

      it('should fail with invalid user id format', async () => {
        const { sessionKey } = await h.createAdminWithSession('admin-delete-invalid-id');

        await h
          .authenticatedRequest(sessionKey)
          .delete(`${s.baseUrl}/users/invalid`)
          .expect('validationError', 'id');
      });
    });

    describe('Authentication Failures', () => {
      it('should fail without authentication', async () => {
        await a.validateUnauthenticatedRejection('/users/1', 'DELETE');
      });

      it('should fail with non-admin user', async () => {
        const { sessionKey } = await h.createUserWithSession('regular-user');
        const targetUser = await h.createUser('unauthorized-delete');

        await spec()
          .delete(`${s.baseUrl}/users/${targetUser.id}`)
          .withCookies(`$S{${sessionKey}}`)
          .expect('authFailure');
      });
    });
  });
});
