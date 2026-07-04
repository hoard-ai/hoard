import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import * as argon from 'argon2';

import { PasswordService } from './password.service';

jest.mock('argon2');
const mockArgon = argon as jest.Mocked<typeof argon>;

describe('PasswordService', () => {
  let passwordService: PasswordService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [PasswordService],
    }).compile();

    passwordService = module.get<PasswordService>(PasswordService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(passwordService).toBeDefined();
  });

  describe('hash', () => {
    it('should hash a password', async () => {
      const password = 'password123';
      const hashedPassword = '$argon2id$v=19$m=65536,t=3,p=4$...';

      mockArgon.hash.mockResolvedValueOnce(hashedPassword);

      const result = await passwordService.hash(password);

      expect(mockArgon.hash).toHaveBeenCalledWith(password);
      expect(mockArgon.hash).toHaveBeenCalledTimes(1);
      expect(result).toBe(hashedPassword);
    });
  });

  describe('verify', () => {
    it('should verify a password successfully when it matches', async () => {
      const hash = '$argon2id$v=19$m=65536,t=3,p=4$...';
      const password = 'password123';

      mockArgon.verify.mockResolvedValueOnce(true);

      const result = await passwordService.verify(hash, password);

      expect(mockArgon.verify).toHaveBeenCalledWith(hash, password);
      expect(mockArgon.verify).toHaveBeenCalledTimes(1);
      expect(result).toBe(true);
    });

    it('should return false when password does not match', async () => {
      const hash = '$argon2id$v=19$m=65536,t=3,p=4$...';
      const password = 'wrongPassword';

      mockArgon.verify.mockResolvedValueOnce(false);

      const result = await passwordService.verify(hash, password);

      expect(mockArgon.verify).toHaveBeenCalledWith(hash, password);
      expect(mockArgon.verify).toHaveBeenCalledTimes(1);
      expect(result).toBe(false);
    });
  });
});
