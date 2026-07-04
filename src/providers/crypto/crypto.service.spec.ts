import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import { CryptoConfigService } from '@/config/crypto';

import { CryptoService } from './crypto.service';

const TEST_KEY = Buffer.from(
  'a6ddee6cf36b3fac8e15a30d9816d08cc80dca9cd1a95a28d834673b5e2e4444',
  'hex',
);

describe('CryptoService', () => {
  let cryptoService: CryptoService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CryptoService,
        {
          provide: CryptoConfigService,
          useValue: { encryptionKey: TEST_KEY },
        },
      ],
    }).compile();

    cryptoService = module.get<CryptoService>(CryptoService);
  });

  it('should be defined', () => {
    expect(cryptoService).toBeDefined();
  });

  describe('encrypt / decrypt', () => {
    it('should round-trip a plaintext string', () => {
      const plaintext = 'sk-ant-api03-secret-key-value';
      const encrypted = cryptoService.encrypt(plaintext);

      expect(encrypted).not.toBe(plaintext);
      expect(cryptoService.decrypt(encrypted)).toBe(plaintext);
    });

    it('should produce different ciphertexts for the same input', () => {
      const plaintext = 'same-key-twice';
      const a = cryptoService.encrypt(plaintext);
      const b = cryptoService.encrypt(plaintext);

      expect(a).not.toBe(b);
    });

    it('should handle empty string', () => {
      const encrypted = cryptoService.encrypt('');
      expect(cryptoService.decrypt(encrypted)).toBe('');
    });

    it('should handle unicode content', () => {
      const plaintext = 'key-with-emoji-🔑-and-中文';
      const encrypted = cryptoService.encrypt(plaintext);
      expect(cryptoService.decrypt(encrypted)).toBe(plaintext);
    });

    it('should throw on tampered ciphertext', () => {
      const encrypted = cryptoService.encrypt('some-secret');
      const tampered =
        encrypted.slice(0, -2) +
        (encrypted.slice(-2, -1) === 'A' ? 'B' : 'A') +
        encrypted.slice(-1);

      expect(() => cryptoService.decrypt(tampered)).toThrow();
    });
  });
});
