import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import type { Session as ExpressSession } from 'express-session';

import { AuthService } from '@/auth/auth.service';
import {
  TEST_USER_UUID,
  TEST_USER_UUID_2,
  UserDtoFactory,
  UserFactory,
} from '@/test/factories';

import type {
  GetUserParamsDto,
  GetUsersQueryDto,
  PaginatedUsers,
  UpdateUserDto,
} from './dto';
import { RoleSchema } from './dto';
import { UserController } from './user.controller';
import { UserService } from './user.service';

describe('UserController', () => {
  let userController: UserController;
  let userService: DeepMocked<UserService>;
  let authService: DeepMocked<AuthService>;
  let mockSession: DeepMocked<ExpressSession>;

  const mockUser = UserFactory.createUserWithoutPassword();

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserController],
    })
      .useMocker(createMock)
      .compile();

    userController = module.get(UserController);
    userService = module.get(UserService);
    authService = module.get(AuthService);
    mockSession = createMock<ExpressSession>();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(userController).toBeDefined();
  });

  describe('getMe', () => {
    it('should return the current user', () => {
      const result = userController.getMe(mockUser);

      expect(result).toEqual(mockUser);
    });
  });

  describe('updateMe', () => {
    it('should call userService.updateSelf with correct parameters', async () => {
      const updateSelfDto = UserDtoFactory.createUpdateSelfDto();
      userService.updateSelf.mockResolvedValueOnce(mockUser);
      const result = await userController.updateMe(mockUser, updateSelfDto);

      expect(userService.updateSelf).toHaveBeenCalledWith(mockUser.id, updateSelfDto);
      expect(userService.updateSelf).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockUser);
    });

    it('should handle update errors', async () => {
      const updateSelfDto = UserDtoFactory.createUpdateSelfDto();
      const error = new Error('Update failed');
      userService.updateSelf.mockRejectedValueOnce(error);

      await expect(userController.updateMe(mockUser, updateSelfDto)).rejects.toThrow(
        'Update failed',
      );
    });
  });

  describe('deleteMe', () => {
    it('should call userService.deleteSelf and authService.destroyUserSession', async () => {
      const deleteSelfDto = UserDtoFactory.createDeleteSelfDto();
      userService.deleteSelf.mockResolvedValueOnce(undefined);
      authService.destroyUserSession.mockResolvedValueOnce(undefined);

      const result = await userController.deleteMe(mockUser, deleteSelfDto, mockSession);
      expect(userService.deleteSelf).toHaveBeenCalledWith(mockUser.id, deleteSelfDto);
      expect(userService.deleteSelf).toHaveBeenCalledTimes(1);

      expect(authService.destroyUserSession).toHaveBeenCalledWith(mockSession);
      expect(authService.destroyUserSession).toHaveBeenCalledTimes(1);
      expect(result).toBeUndefined();
    });

    it('should handle user deletion errors', async () => {
      const deleteSelfDto = UserDtoFactory.createDeleteSelfDto();
      const error = new Error('Deletion failed');
      userService.deleteSelf.mockRejectedValueOnce(error);

      await expect(
        userController.deleteMe(mockUser, deleteSelfDto, mockSession),
      ).rejects.toThrow('Deletion failed');

      expect(authService.destroyUserSession).not.toHaveBeenCalled();
    });

    it('should not fail if session destruction fails', async () => {
      const deleteSelfDto = UserDtoFactory.createDeleteSelfDto();

      userService.deleteSelf.mockResolvedValueOnce(undefined);
      authService.destroyUserSession.mockRejectedValueOnce(new Error('Session error'));
      const loggerSpy = jest
        .spyOn(userController['logger'], 'error')
        .mockImplementation(() => {});

      await expect(
        userController.deleteMe(mockUser, deleteSelfDto, mockSession),
      ).resolves.toBeUndefined();
      expect(loggerSpy).toHaveBeenCalledWith(
        'Failed to destroy session after account deletion',
        expect.any(Error),
      );
    });
  });

  describe('createUser ( only)', () => {
    it('should call userService.createUserWithDto with correct parameters', async () => {
      const createUserDto = UserDtoFactory.createCreateUserDto();
      const createUserResult = UserFactory.createUserWithoutPassword({
        email: createUserDto.email,
        role: createUserDto.role,
      });
      userService.createUserWithDto.mockResolvedValueOnce(createUserResult);

      const result = await userController.createUser(createUserDto);

      expect(userService.createUserWithDto).toHaveBeenCalledWith(createUserDto);
      expect(userService.createUserWithDto).toHaveBeenCalledTimes(1);
      expect(result).toEqual(createUserResult);
    });

    it('should handle creation errors', async () => {
      const createUserDto = UserDtoFactory.createCreateUserDto();
      const error = new Error('Creation failed');
      userService.createUserWithDto.mockRejectedValueOnce(error);

      await expect(userController.createUser(createUserDto)).rejects.toThrow(
        'Creation failed',
      );
    });
  });

  describe('getAllUsers (Admin only)', () => {
    it('should call userService.getAllUsers with correct query parameters', async () => {
      const query: GetUsersQueryDto = {
        page: 1,
        limit: 10,
        search: 'test',
      };

      const mockPaginatedUsers: PaginatedUsers = {
        users: [mockUser],
        pagination: {
          page: 1,
          limit: 10,
          total: 1,
          totalPages: 1,
          hasNext: false,
          hasPrev: false,
        },
      };

      userService.getAllUsers.mockResolvedValueOnce(mockPaginatedUsers);

      const result = await userController.getAllUsers(query);

      expect(userService.getAllUsers).toHaveBeenCalledWith(query);
      expect(userService.getAllUsers).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockPaginatedUsers);
    });

    it('should handle query errors', async () => {
      const query: GetUsersQueryDto = {
        page: 1,
        limit: 10,
      };
      const error = new Error('Query failed');
      userService.getAllUsers.mockRejectedValueOnce(error);

      await expect(userController.getAllUsers(query)).rejects.toThrow('Query failed');
    });
  });

  describe('getUserById (Admin only)', () => {
    it('should call userService.getUserById with correct parameters', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID };
      const mockFullUser = UserFactory.createUser();

      userService.getUserById.mockResolvedValueOnce(mockFullUser);

      const result = await userController.getUserById(params);

      expect(userService.getUserById).toHaveBeenCalledWith(params.id);
      expect(userService.getUserById).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockFullUser);
    });

    it('should handle user not found errors', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID_2 };
      const error = new Error('User not found');
      userService.getUserById.mockRejectedValueOnce(error);

      await expect(userController.getUserById(params)).rejects.toThrow('User not found');
    });
  });

  describe('updateUser (Admin only)', () => {
    it('should call userService.updateUser with correct parameters', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID };
      const updateUserDto: UpdateUserDto = {
        email: 'updated@example.com',
        role: RoleSchema.enum.ADMIN,
      };
      userService.updateUser.mockResolvedValueOnce(mockUser);

      const result = await userController.updateUser(params, updateUserDto);

      expect(userService.updateUser).toHaveBeenCalledWith(params.id, updateUserDto);
      expect(userService.updateUser).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockUser);
    });

    it('should handle update errors', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID };
      const updateUserDto: UpdateUserDto = {
        email: 'updated@example.com',
      };
      const error = new Error('Update failed');
      userService.updateUser.mockRejectedValueOnce(error);

      await expect(userController.updateUser(params, updateUserDto)).rejects.toThrow(
        'Update failed',
      );
    });
  });

  describe('deleteUser (Admin only)', () => {
    it('should call userService.deleteUser with correct parameters', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID };

      userService.deleteUser.mockResolvedValueOnce(undefined);

      const result = await userController.deleteUser(params);

      expect(userService.deleteUser).toHaveBeenCalledWith(params.id);
      expect(userService.deleteUser).toHaveBeenCalledTimes(1);
      expect(result).toBeUndefined();
    });

    it('should handle deletion errors', async () => {
      const params: GetUserParamsDto = { id: TEST_USER_UUID };
      const error = new Error('Deletion failed');
      userService.deleteUser.mockRejectedValueOnce(error);

      await expect(userController.deleteUser(params)).rejects.toThrow('Deletion failed');
    });
  });
});
