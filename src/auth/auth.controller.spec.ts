import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import { InternalServerErrorException } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import type { Request } from 'express';

import { AuthFactory, UserFactory } from '@/test/factories';

import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';

describe('AuthController', () => {
  let authController: AuthController;
  let authService: DeepMocked<AuthService>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [AuthController],
    })
      .useMocker(createMock)
      .compile();

    authController = module.get(AuthController);
    authService = module.get(AuthService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(authController).toBeDefined();
  });

  describe('registerUser', () => {
    it('should call authService.registerUser with correct parameters', async () => {
      const authDto = AuthFactory.createAuthDto();

      const expectedResult = {
        message: 'Registration successful',
        user: UserFactory.createAuthServiceResult(),
      };
      authService.registerUser.mockResolvedValueOnce(expectedResult.user);

      const result = await authController.registerUser(authDto);

      expect(authService.registerUser).toHaveBeenCalledWith(authDto);
      expect(authService.registerUser).toHaveBeenCalledTimes(1);
      expect(result).toEqual(expectedResult);
    });

    it('should handle registration errors', async () => {
      const authDto = AuthFactory.createAuthDto();
      const error = new Error('Registration failed');
      authService.registerUser.mockRejectedValueOnce(error);

      await expect(authController.registerUser(authDto)).rejects.toThrow(
        'Registration failed',
      );
    });
  });

  describe('loginUser', () => {
    it('should return login success message with user data', () => {
      const mockUserWithoutPassword = UserFactory.createUserWithoutPassword();
      const result = authController.loginUser(mockUserWithoutPassword);

      expect(result).toEqual({
        message: 'Login successful',
        user: mockUserWithoutPassword,
      });
    });
  });

  describe('logoutUser', () => {
    it('should successfully logout user', async () => {
      const mockLogout = jest.fn<void, [(err: any) => void]>((callback) =>
        callback(null),
      );

      const mockRequest = createMock<Request>({
        logout: mockLogout,
      });
      const result = await authController.logoutUser(mockRequest);

      expect(mockLogout).toHaveBeenCalledTimes(1);
      expect(result).toEqual({ message: 'Logout successful' });
    });

    it('should handle logout errors', async () => {
      const mockLogout = jest.fn<void, [(err: any) => void]>((callback) =>
        callback(new Error('Logout error')),
      );
      const mockRequest = createMock<Request>({
        logout: mockLogout,
      });

      await expect(authController.logoutUser(mockRequest)).rejects.toThrow(
        InternalServerErrorException,
      );
      await expect(authController.logoutUser(mockRequest)).rejects.toThrow(
        'Logout failed',
      );
    });
  });
});
