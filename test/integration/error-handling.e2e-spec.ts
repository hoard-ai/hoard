import { spec } from 'pactum';
import type { ExpectHandlerContext } from 'pactum/src/exports/handler';

import {
  TestAssertions as a,
  TestDataFactory as f,
  TestHelpers as h,
  TestSetup as s,
} from '@test/setup';

describe('Error Handling Integration (e2e)', () => {
  describe('Malformed Request Handling', () => {
    it('should handle malformed requests gracefully', async () => {
      const malformedScenarios = f.createMalformedScenarios();

      const endpoints = [
        { url: `${s.baseUrl}/auth/signup` },
        { url: `${s.baseUrl}/auth/login` },
      ];

      for (const endpoint of endpoints) {
        for (const scenario of malformedScenarios) {
          await spec()
            .post(endpoint.url)
            .withHeaders('Content-Type', scenario.contentType)
            .withBody(scenario.body)
            .expectBodyContains('statusCode')
            .expect((r: ExpectHandlerContext) =>
              [400, 415].includes((r.res.body as { statusCode: number }).statusCode),
            );
        }
      }
    });

    it('should handle missing content-type header', async () => {
      const validData = f.createUserCredentials('content-type');

      await spec()
        .post(`${s.baseUrl}/auth/signup`)
        .withHeaders('Content-Type', 'text/plain')
        .withBody(JSON.stringify(validData))
        .expectStatus(400);
    });

    it('should handle extremely large payloads gracefully', async () => {
      const largeEmail = 'x'.repeat(1000) + '@example.com';
      const largePassword = 'x'.repeat(1000);

      await spec()
        .post(`${s.baseUrl}/auth/signup`)
        .withBody({
          email: largeEmail,
          password: largePassword,
        })
        .expectStatus(400);
    });
  });

  describe('Rapid Request Handling', () => {
    it('should handle rapid successive operations gracefully', async () => {
      const { sessionKey } = await h.createUserWithSession('rapid-operations');

      const rapidRequests = Array.from({ length: 10 }, () =>
        h
          .authenticatedRequest(sessionKey)
          .get(`${s.baseUrl}/users/me`)
          .expect('validUserResponse')
          .toss(),
      );
      await Promise.all(rapidRequests);
    });

    it('should handle rapid login attempts', async () => {
      const { credentials } = await h.createUser('rapid-login');

      const loginRequests = Array.from({ length: 5 }, () =>
        spec()
          .post(`${s.baseUrl}/auth/login`)
          .withBody(credentials)
          .expectStatus(200)
          .toss(),
      );
      await Promise.all(loginRequests);
    });

    it('should handle rapid profile updates', async () => {
      const { sessionKey, credentials } = await h.createUserWithSession('rapid-updates');

      const updateScenarios = [
        f.createUpdateScenarios(credentials).emailOnly(),
        f.createUpdateScenarios(credentials).passwordOnly(),
      ];
      for (const updateData of updateScenarios) {
        await h
          .authenticatedRequest(sessionKey)
          .put(`${s.baseUrl}/users/me`)
          .withBody(updateData)
          .expectStatus(200);
      }
    });
  });

  describe('Network and Timeout Simulation', () => {
    it('should handle multiple simultaneous signups', async () => {
      const userPromises = Array.from({ length: 5 }, (_, index) =>
        spec()
          .post(`${s.baseUrl}/auth/signup`)
          .withBody(f.createUserCredentials(`simultaneous-${index}`))
          .expectStatus(201)
          .toss(),
      );

      await Promise.all(userPromises);
    });

    it('should handle session operations under load', async () => {
      const users = await Promise.all(
        Array.from({ length: 6 }, (_, index) =>
          h.createUserWithSession(`load-test-${index}`),
        ),
      );

      const operations = users.flatMap((user) => [
        a.validateAuthenticatedAccess(user.sessionKey),
        h
          .authenticatedRequest(user.sessionKey)
          .put(`${s.baseUrl}/users/me`)
          .withBody(f.createUpdateScenarios(user.credentials).emailOnly())
          .expectStatus(200)
          .toss(),
        a.validateAuthenticatedAccess(user.sessionKey),
      ]);

      await Promise.all(operations);
    });
  });

  describe('Resource Cleanup Edge Cases', () => {
    it('should handle account deletion during active session', async () => {
      const { credentials } = await h.createUser('deletion-edge');

      const sessions = await Promise.all([
        h.loginUser(credentials),
        h.loginUser(credentials),
      ]);

      // Delete account using first session
      const deleteData = f.createValidDeleteData(credentials.password);
      await h
        .authenticatedRequest(sessions[0].sessionKey)
        .delete(`${s.baseUrl}/users/me`)
        .withBody(deleteData)
        .expectStatus(204);

      // All sessions should be invalidated
      await Promise.all(
        sessions.map((session) =>
          h
            .authenticatedRequest(session.sessionKey)
            .get(`${s.baseUrl}/users/me`)
            .expect('authFailure')
            .toss(),
        ),
      );
    });

    it('should handle logout during concurrent operations', async () => {
      const { sessionKey, credentials } =
        await h.createUserWithSession('logout-concurrent');

      // Start a profile update
      const updatePromise = h
        .authenticatedRequest(sessionKey)
        .put(`${s.baseUrl}/users/me`)
        .withBody(f.createUpdateScenarios(credentials).emailOnly())
        .toss();

      // Immediately logout (simulate race condition)
      const logoutPromise = h.logoutUser(sessionKey);

      // Wait for both operations
      await Promise.all([updatePromise, logoutPromise]);

      // Session should definitely be invalid after logout
      await h
        .authenticatedRequest(sessionKey)
        .get(`${s.baseUrl}/users/me`)
        .expect('authFailure');
    });
  });
});
