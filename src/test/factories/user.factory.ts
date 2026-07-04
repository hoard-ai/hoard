import type { Graph as PrismaGraph, User as PrismaUser } from '@generated/prisma/client';
import { Role as PrismaRole } from '@generated/prisma/client';

import type { Uuid } from '@/common/schemas';
import { UuidSchema } from '@/common/schemas';
import type { Role, User, UserWithoutPassword } from '@/user/dto';
import { RoleSchema } from '@/user/dto';

export type PrismaUserWithGraphs = PrismaUser & { graphs: PrismaGraph[] };

export const TEST_USER_UUID = UuidSchema.parse('00000000-0000-4000-8000-000000000001');
export const TEST_USER_UUID_2 = UuidSchema.parse('00000000-0000-4000-8000-000000000002');

export interface UserFactoryOptions {
  id?: Uuid;
  email?: string;
  role?: Role;
  createdAt?: string;
  updatedAt?: string;
  passwordHash?: string;
}

export interface PrismaUserFactoryOptions {
  id?: string;
  email?: string;
  role?: PrismaRole;
  createdAt?: Date;
  updatedAt?: Date;
  passwordHash?: string;
}

export class UserFactory {
  private static defaultUser: User = {
    id: TEST_USER_UUID,
    email: 'test@example.com',
    role: RoleSchema.enum.USER,
    activeLlmProvider: null,
    createdAt: new Date('2025-01-01').toISOString(),
    updatedAt: new Date('2025-01-01').toISOString(),
    passwordHash: '$argon2id$v=19$m=65536,t=3,p=4$...',
    graphs: [],
  };

  private static defaultPrismaUser: PrismaUserWithGraphs = {
    id: TEST_USER_UUID,
    email: 'test@example.com',
    role: PrismaRole.USER,
    activeLlmProvider: null,
    createdAt: new Date('2025-01-01'),
    updatedAt: new Date('2025-01-01'),
    passwordHash: '$argon2id$v=19$m=65536,t=3,p=4$...',
    graphs: [],
  };

  /**
   * Creates a full User object (including passwordHash)
   */
  static createUser(options: UserFactoryOptions = {}): User {
    return {
      ...this.defaultUser,
      ...options,
    };
  }

  /**
   * Creates a UserWithoutPassword object (excludes passwordHash)
   */
  static createUserWithoutPassword(
    options: UserFactoryOptions = {},
  ): UserWithoutPassword {
    const user = this.createUser(options);
    const { passwordHash: _, ...userWithoutPassword } = user;
    return userWithoutPassword;
  }

  /**
   * Creates a user service return type (what createUser method returns)
   */
  static createUserServiceResult(options: Omit<UserFactoryOptions, 'passwordHash'> = {}) {
    const { passwordHash: _, ...userOptions } = this.defaultUser;
    return {
      ...userOptions,
      ...options,
    };
  }

  /**
   * Creates an auth service return type (what registerUser/validateUser returns)
   */
  static createAuthServiceResult(options: Omit<UserFactoryOptions, 'passwordHash'> = {}) {
    const { passwordHash: _, ...userOptions } = this.defaultUser;
    return {
      ...userOptions,
      ...options,
    };
  }

  /**
   * Creates a full Prisma User object (including passwordHash) with Date objects
   * Use this for mocking Prisma service calls
   */
  static createPrismaUser(options: PrismaUserFactoryOptions = {}): PrismaUserWithGraphs {
    return {
      ...this.defaultPrismaUser,
      ...options,
    };
  }

  /**
   * Creates a Prisma UserWithoutPassword object (excludes passwordHash) with Date objects
   * Use this for mocking Prisma service calls that select without passwordHash
   */
  static createPrismaUserWithoutPassword(
    options: PrismaUserFactoryOptions = {},
  ): Omit<PrismaUserWithGraphs, 'passwordHash'> {
    const user = this.createPrismaUser(options);
    const { passwordHash: _, ...userWithoutPassword } = user;
    return userWithoutPassword;
  }
}
