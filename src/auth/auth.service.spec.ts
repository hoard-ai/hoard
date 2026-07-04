import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import { ForbiddenException, NotFoundException } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import type { Session } from 'express-session';

import { AuthFactory, UserFactory } from '@/test/factories';
import { UserService } from '@/user/user.service';

import { AuthService } from './auth.service';
import { PasswordService } from './password.service';

describe('AuthService', () => {
  let authService: AuthService;
  let userService: DeepMocked<UserService>;
  let passwordService: DeepMocked<PasswordService>;
  let mockSession: DeepMocked<Session>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [AuthService],
    })
      .useMocker(createMock)
      .compile();

    authService = module.get(AuthService);
    userService = module.get(UserService);
    passwordService = module.get(PasswordService);
    mockSession = createMock<Session>();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(authService).toBeDefined();
  });

  describe('registerUser', () => {
    const authDto = AuthFactory.createAuthDto();

    it('should call userService.createUser with a hashed password', async () => {
      const hashedPassword = '$argon2id$v=19$m=65536,t=3,p=4$...';
      passwordService.hash.mockResolvedValueOnce(hashedPassword);

      const userServiceResult = UserFactory.createUserServiceResult();
      userService.createUser.mockResolvedValueOnce(userServiceResult);

      const expectedResult = UserFactory.createAuthServiceResult();
      const result = await authService.registerUser(authDto);

      expect(passwordService.hash).toHaveBeenCalledWith(authDto.password);
      expect(passwordService.hash).toHaveBeenCalledTimes(1);

      expect(userService.createUser).toHaveBeenCalledWith(authDto.email, hashedPassword);
      expect(userService.createUser).toHaveBeenCalledTimes(1);
      expect(result).toEqual(expectedResult);
    });
  });

  describe('validateUser', () => {
    const authDto = AuthFactory.createAuthDto();
    const mockUser = UserFactory.createUser();

    it('should return user data when credentials are valid', async () => {
      userService.getUserByEmail.mockResolvedValueOnce(mockUser);
      passwordService.verify.mockResolvedValueOnce(true);

      const result = await authService.validateUser(authDto);

      expect(userService.getUserByEmail).toHaveBeenCalledWith(authDto.email);
      expect(userService.getUserByEmail).toHaveBeenCalledTimes(1);

      expect(passwordService.verify).toHaveBeenCalledWith(
        mockUser.passwordHash,
        authDto.password,
      );
      expect(passwordService.verify).toHaveBeenCalledTimes(1);

      expect(result).toEqual(UserFactory.createAuthServiceResult());
    });

    it('should throw ForbiddenException when getUserByEmail throws NotFoundException', async () => {
      userService.getUserByEmail.mockRejectedValueOnce(
        new NotFoundException('User not found'),
      );

      await expect(authService.validateUser(authDto)).rejects.toThrow(
        new ForbiddenException('Invalid credentials'),
      );

      expect(userService.getUserByEmail).toHaveBeenCalledWith(authDto.email);
      expect(userService.getUserByEmail).toHaveBeenCalledTimes(1);

      expect(passwordService.verify).not.toHaveBeenCalled();
    });

    it('should throw ForbiddenException when password is incorrect', async () => {
      userService.getUserByEmail.mockResolvedValueOnce(mockUser);
      passwordService.verify.mockResolvedValueOnce(false);

      await expect(authService.validateUser(authDto)).rejects.toThrow(
        new ForbiddenException('Invalid credentials'),
      );

      expect(userService.getUserByEmail).toHaveBeenCalledWith(authDto.email);
      expect(userService.getUserByEmail).toHaveBeenCalledTimes(1);

      expect(passwordService.verify).toHaveBeenCalledWith(
        mockUser.passwordHash,
        authDto.password,
      );
      expect(passwordService.verify).toHaveBeenCalledTimes(1);
    });

    it('should re-throw non-NotFoundException errors', async () => {
      const unexpectedError = new Error('Database connection failed');
      userService.getUserByEmail.mockRejectedValueOnce(unexpectedError);

      await expect(authService.validateUser(authDto)).rejects.toThrow(unexpectedError);

      expect(userService.getUserByEmail).toHaveBeenCalledWith(authDto.email);
      expect(passwordService.verify).not.toHaveBeenCalled();
    });
  });

  describe('destroyUserSession', () => {
    it('should successfully destroy session', async () => {
      mockSession.destroy.mockImplementation((callback) => {
        callback(undefined);
        return mockSession;
      });

      await expect(authService.destroyUserSession(mockSession)).resolves.toBeUndefined();
      expect(mockSession.destroy).toHaveBeenCalledTimes(1);
    });

    it('should handle session destruction errors gracefully', async () => {
      const sessionError = new Error('Session destruction failed');
      mockSession.destroy.mockImplementation((callback) => {
        callback(sessionError);
        return mockSession;
      });

      const loggerSpy = jest.spyOn(authService['logger'], 'error').mockImplementation();

      await expect(authService.destroyUserSession(mockSession)).resolves.toBeUndefined();
      expect(loggerSpy).toHaveBeenCalledWith(
        'Failed to destroy user session',
        sessionError,
      );
    });
  });
});
