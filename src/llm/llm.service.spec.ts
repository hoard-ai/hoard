import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { BadRequestException, NotFoundException } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';
import type { DeepMockProxy } from 'jest-mock-extended';
import { mockDeep, mockReset } from 'jest-mock-extended';

import { LlmProvider } from '@generated/prisma/client';

import { LlmConfigService } from '@/config/llm';
import { CryptoService } from '@/providers/crypto';
import { PrismaService } from '@/providers/database/postgres';
import { RateLimiterService } from '@/providers/rate-limit';
import { TEST_USER_UUID, TestLlmFactory, UserFactory } from '@/test/factories';

import { UserLlmProviderSchema } from './dto';
import { LlmFactoryService } from './factory/llm-factory.service';
import { LlmService } from './llm.service';
import { DEFAULT_MODEL_CONCURRENCY } from './types';

describe('LlmService', () => {
  let llmService: LlmService;
  let prismaService: DeepMockProxy<PrismaService>;
  let factoryService: DeepMocked<LlmFactoryService>;
  let llmConfigService: DeepMocked<LlmConfigService>;
  let cryptoService: DeepMocked<CryptoService>;
  let rateLimiterService: DeepMocked<RateLimiterService>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        LlmService,
        { provide: PrismaService, useValue: mockDeep<PrismaService>() },
        {
          provide: LlmFactoryService,
          useValue: createMock<LlmFactoryService>(),
        },
        {
          provide: LlmConfigService,
          useValue: createMock<LlmConfigService>(),
        },
        { provide: CryptoService, useValue: createMock<CryptoService>() },
        {
          provide: RateLimiterService,
          useValue: createMock<RateLimiterService>(),
        },
      ],
    }).compile();

    llmService = module.get(LlmService);
    prismaService = module.get(PrismaService);
    factoryService = module.get(LlmFactoryService);
    llmConfigService = module.get(LlmConfigService);
    cryptoService = module.get(CryptoService);
    rateLimiterService = module.get(RateLimiterService);
  });

  afterEach(() => {
    jest.clearAllMocks();
    mockReset(prismaService);
  });

  it('should be defined', () => {
    expect(llmService).toBeDefined();
  });

  describe('listProviders', () => {
    it('should return exactly one entry per value in UserLlmProviderSchema.options', async () => {
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );
      prismaService.llmConfig.findMany.mockResolvedValueOnce([]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      const expectedProviders = new Set(UserLlmProviderSchema.options);
      expect(result.providers).toHaveLength(expectedProviders.size);

      for (const provider of expectedProviders) {
        expect(result.providers.some((p) => p.provider === provider)).toBe(true);
      }
    });

    it('should never include PLATFORM in providers list', async () => {
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );
      prismaService.llmConfig.findMany.mockResolvedValueOnce([]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      expect(result.providers.map((p) => p.provider)).not.toContain(LlmProvider.PLATFORM);
    });

    it('should correctly map hasApiKey: true when config blob has apiKey', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig({
        provider: LlmProvider.ANTHROPIC,
      });
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );
      prismaService.llmConfig.findMany.mockResolvedValueOnce([mockConfig]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      const anthropicEntry = result.providers.find(
        (p) => p.provider === LlmProvider.ANTHROPIC,
      );
      const geminiEntry = result.providers.find(
        (p) => p.provider === LlmProvider.GOOGLE_GEMINI,
      );
      expect(anthropicEntry?.hasApiKey).toBe(true);
      expect(geminiEntry?.hasApiKey).toBe(false);
    });

    it('should correctly map hasApiKey: false when config row has no apiKey', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfigWithoutApiKey({
        provider: LlmProvider.ANTHROPIC,
      });
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );
      prismaService.llmConfig.findMany.mockResolvedValueOnce([mockConfig]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      const anthropicEntry = result.providers.find(
        (p) => p.provider === LlmProvider.ANTHROPIC,
      );
      expect(anthropicEntry?.hasApiKey).toBe(false);
    });

    it('should reflect activeLlmProvider from DB', async () => {
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce({
        ...UserFactory.createPrismaUser(),
        activeLlmProvider: LlmProvider.ANTHROPIC,
      });
      prismaService.llmConfig.findMany.mockResolvedValueOnce([]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      expect(result.activeProvider).toBe(LlmProvider.ANTHROPIC);
    });

    it('should return activeProvider: null when user has none', async () => {
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );
      prismaService.llmConfig.findMany.mockResolvedValueOnce([]);

      const result = await llmService.listProviders(TEST_USER_UUID);

      expect(result.activeProvider).toBeNull();
    });
  });

  describe('getProviderConfig', () => {
    it('should throw BadRequestException for PLATFORM', async () => {
      await expect(
        llmService.getProviderConfig(TEST_USER_UUID, LlmProvider.PLATFORM),
      ).rejects.toThrow(new BadRequestException('Platform provider is server-managed'));
    });

    it('should return { hasApiKey: false } shell when no row exists', async () => {
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);

      const result = await llmService.getProviderConfig(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(result.provider).toBe(LlmProvider.ANTHROPIC);
      expect(result.hasApiKey).toBe(false);
    });

    it('should return full response with hasApiKey: true when row exists with apiKey', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);

      const result = await llmService.getProviderConfig(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(result.provider).toBe(LlmProvider.ANTHROPIC);
      expect(result.hasApiKey).toBe(true);
      expect(result.model).toBeDefined();
    });
  });

  describe('saveProviderConfig', () => {
    it('should encrypt the apiKey when one is provided', async () => {
      const saveConfig = TestLlmFactory.createSaveLlmConfig({
        apiKey: 'plain-key',
      });
      const mockResult = TestLlmFactory.createPrismaLlmConfig();

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);
      cryptoService.encrypt.mockReturnValueOnce('ENCRYPTED_KEY');
      prismaService.llmConfig.upsert.mockResolvedValueOnce(mockResult);

      await llmService.saveProviderConfig(TEST_USER_UUID, saveConfig);

      expect(cryptoService.encrypt).toHaveBeenCalledWith('plain-key');
      expect(prismaService.llmConfig.upsert).toHaveBeenCalledWith(
        expect.objectContaining({
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          create: expect.objectContaining({
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            config: expect.objectContaining({ apiKey: 'ENCRYPTED_KEY' }),
          }),
        }),
      );
    });

    it('should upsert with update clause when row already exists', async () => {
      const existingRow = TestLlmFactory.createPrismaLlmConfig();
      const saveConfig = TestLlmFactory.createSaveLlmConfig({
        model: 'claude-3-opus-20240229',
      });
      const mockResult = TestLlmFactory.createPrismaLlmConfig({
        model: 'claude-3-opus-20240229',
      });

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(existingRow);
      cryptoService.encrypt.mockReturnValueOnce('ENCRYPTED_KEY');
      prismaService.llmConfig.upsert.mockResolvedValueOnce(mockResult);

      await llmService.saveProviderConfig(TEST_USER_UUID, saveConfig);

      expect(prismaService.llmConfig.upsert).toHaveBeenCalledWith(
        expect.objectContaining({
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          update: expect.objectContaining({
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            config: expect.objectContaining({
              apiKey: 'ENCRYPTED_KEY',
              model: 'claude-3-opus-20240229',
            }),
          }),
        }),
      );
    });

    it('should retain existing encrypted key when incoming has no apiKey', async () => {
      const existingRow = TestLlmFactory.createPrismaLlmConfig({
        encryptedApiKey: 'EXISTING_ENCRYPTED',
      });
      const saveConfig = TestLlmFactory.createSaveLlmConfig({
        apiKey: undefined,
      });
      const mockResult = TestLlmFactory.createPrismaLlmConfig();

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(existingRow);
      prismaService.llmConfig.upsert.mockResolvedValueOnce(mockResult);

      await llmService.saveProviderConfig(TEST_USER_UUID, saveConfig);

      expect(cryptoService.encrypt).not.toHaveBeenCalled();
      // The upsert config should carry the existing encrypted key
      expect(prismaService.llmConfig.upsert).toHaveBeenCalledWith(
        expect.objectContaining({
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          update: expect.objectContaining({
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            config: expect.objectContaining({ apiKey: 'EXISTING_ENCRYPTED' }),
          }),
        }),
      );
    });

    it('should encrypt new key and discard old when a new apiKey is provided', async () => {
      const existingRow = TestLlmFactory.createPrismaLlmConfig({
        encryptedApiKey: 'OLD_ENCRYPTED',
      });
      const saveConfig = TestLlmFactory.createSaveLlmConfig({
        apiKey: 'new-plain-key',
      });
      const mockResult = TestLlmFactory.createPrismaLlmConfig();

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(existingRow);
      cryptoService.encrypt.mockReturnValueOnce('NEW_ENCRYPTED');
      prismaService.llmConfig.upsert.mockResolvedValueOnce(mockResult);

      await llmService.saveProviderConfig(TEST_USER_UUID, saveConfig);

      expect(cryptoService.encrypt).toHaveBeenCalledWith('new-plain-key');
      expect(prismaService.llmConfig.upsert).toHaveBeenCalledWith(
        expect.objectContaining({
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          update: expect.objectContaining({
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            config: expect.objectContaining({ apiKey: 'NEW_ENCRYPTED' }),
          }),
        }),
      );
    });

    it('should create config without apiKey field when none ever provided', async () => {
      const saveConfig = TestLlmFactory.createSaveLlmConfig({
        apiKey: undefined,
      });
      const mockResult = TestLlmFactory.createPrismaLlmConfigWithoutApiKey();

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);
      prismaService.llmConfig.upsert.mockResolvedValueOnce(mockResult);

      await llmService.saveProviderConfig(TEST_USER_UUID, saveConfig);

      expect(cryptoService.encrypt).not.toHaveBeenCalled();
      const upsertCall = prismaService.llmConfig.upsert.mock.calls[0][0];
      expect(upsertCall.create.config as Record<string, unknown>).not.toHaveProperty(
        'apiKey',
      );
    });
  });

  describe('deleteProviderConfig', () => {
    beforeEach(() => {
      (prismaService.$transaction as jest.Mock).mockImplementation(
        (fn: (tx: typeof prismaService) => unknown) => fn(prismaService),
      );
    });

    it('should throw BadRequestException for PLATFORM', async () => {
      await expect(
        llmService.deleteProviderConfig(TEST_USER_UUID, LlmProvider.PLATFORM),
      ).rejects.toThrow(new BadRequestException('Platform provider is server-managed'));
    });

    it('should throw NotFoundException inside transaction when config not found', async () => {
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);

      await expect(
        llmService.deleteProviderConfig(TEST_USER_UUID, LlmProvider.ANTHROPIC),
      ).rejects.toThrow(new NotFoundException('Provider config not found'));
    });

    it('should call delete and updateMany in the same transaction with correct args', async () => {
      const configId = '00000000-0000-4000-8000-000000000042';
      const mockConfig = TestLlmFactory.createPrismaLlmConfig({ id: configId });
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      prismaService.llmConfig.delete.mockResolvedValueOnce(mockConfig);
      prismaService.user.updateMany.mockResolvedValueOnce({ count: 0 });

      await llmService.deleteProviderConfig(TEST_USER_UUID, LlmProvider.ANTHROPIC);

      expect(prismaService.llmConfig.delete).toHaveBeenCalledWith({
        where: { id: configId },
      });
      expect(prismaService.user.updateMany).toHaveBeenCalledWith({
        where: { id: TEST_USER_UUID, activeLlmProvider: LlmProvider.ANTHROPIC },
        data: { activeLlmProvider: null },
      });
    });
  });

  describe('setActiveProvider', () => {
    it('should set null without querying the config table', async () => {
      prismaService.user.update.mockResolvedValueOnce(UserFactory.createPrismaUser());
      const result = await llmService.setActiveProvider(TEST_USER_UUID, null);

      expect(prismaService.llmConfig.findUnique).not.toHaveBeenCalled();
      expect(prismaService.user.update).toHaveBeenCalledWith({
        where: { id: TEST_USER_UUID },
        data: { activeLlmProvider: null },
      });
      expect(result).toEqual({ activeProvider: null });
    });

    it('should throw BadRequestException when PLATFORM and platformModelEnabled is false', async () => {
      Object.defineProperty(llmConfigService, 'platformModelEnabled', {
        get: () => false,
        configurable: true,
      });

      await expect(
        llmService.setActiveProvider(TEST_USER_UUID, LlmProvider.PLATFORM),
      ).rejects.toThrow(new BadRequestException('Platform model is not enabled'));
    });

    it('should succeed setting PLATFORM when platform model is enabled', async () => {
      Object.defineProperty(llmConfigService, 'platformModelEnabled', {
        get: () => true,
        configurable: true,
      });
      prismaService.user.update.mockResolvedValueOnce(UserFactory.createPrismaUser());

      const result = await llmService.setActiveProvider(
        TEST_USER_UUID,
        LlmProvider.PLATFORM,
      );

      expect(result).toEqual({ activeProvider: LlmProvider.PLATFORM });
    });

    it('should throw BadRequestException when user provider has no saved config row', async () => {
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);

      await expect(
        llmService.setActiveProvider(TEST_USER_UUID, LlmProvider.ANTHROPIC),
      ).rejects.toThrow(
        new BadRequestException(
          `Provider ${LlmProvider.ANTHROPIC} is not configured. Save a config first.`,
        ),
      );
    });

    it('should throw BadRequestException when config row has no apiKey in blob', async () => {
      const configWithoutApiKey = TestLlmFactory.createPrismaLlmConfigWithoutApiKey();
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(configWithoutApiKey);

      await expect(
        llmService.setActiveProvider(TEST_USER_UUID, LlmProvider.ANTHROPIC),
      ).rejects.toThrow(
        new BadRequestException(
          `Provider ${LlmProvider.ANTHROPIC} is missing an API key`,
        ),
      );
    });

    it('should succeed and return { activeProvider: provider } when config has apiKey', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      prismaService.user.update.mockResolvedValueOnce(UserFactory.createPrismaUser());

      const result = await llmService.setActiveProvider(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(prismaService.user.update).toHaveBeenCalledWith({
        where: { id: TEST_USER_UUID },
        data: { activeLlmProvider: LlmProvider.ANTHROPIC },
      });
      expect(result).toEqual({ activeProvider: LlmProvider.ANTHROPIC });
    });
  });

  describe('testProviderConfig', () => {
    it('should throw BadRequestException for PLATFORM', async () => {
      await expect(
        llmService.testProviderConfig(TEST_USER_UUID, LlmProvider.PLATFORM),
      ).rejects.toThrow(new BadRequestException('Platform model is not testable'));
    });

    it('should return success result when model.invoke succeeds', async () => {
      const mockModel = {
        invoke: jest.fn().mockResolvedValueOnce('hello'),
      } as unknown as BaseChatModel;
      jest.spyOn(llmService, 'getActiveModel').mockResolvedValueOnce(mockModel);

      const result = await llmService.testProviderConfig(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(result.success).toBe(true);
      expect(result.message).toBe('Connection successful');
      expect(result.responseTime).toEqual(expect.any(Number));
    });

    it('should return failure result when model.invoke throws an Error', async () => {
      const mockModel = {
        invoke: jest.fn().mockRejectedValueOnce(new Error('Auth failed')),
      } as unknown as BaseChatModel;
      jest.spyOn(llmService, 'getActiveModel').mockResolvedValueOnce(mockModel);

      const result = await llmService.testProviderConfig(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(result.success).toBe(false);
      expect(result.message).toBe('Auth failed');
      expect(result.responseTime).toEqual(expect.any(Number));
    });

    it('should use "Unknown error occurred" when thrown value is not an Error instance', async () => {
      const mockModel = {
        invoke: jest.fn().mockRejectedValueOnce('some string error'),
      } as unknown as BaseChatModel;
      jest.spyOn(llmService, 'getActiveModel').mockResolvedValueOnce(mockModel);

      const result = await llmService.testProviderConfig(
        TEST_USER_UUID,
        LlmProvider.ANTHROPIC,
      );

      expect(result.success).toBe(false);
      expect(result.message).toBe('Unknown error occurred');
    });
  });

  describe('getActiveModel', () => {
    it('should throw BadRequestException when activeLlmProvider is null and no override', async () => {
      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce(
        UserFactory.createPrismaUser(),
      );

      await expect(llmService.getActiveModel(TEST_USER_UUID)).rejects.toThrow(
        new BadRequestException('No active LLM provider is set'),
      );
    });

    it('should use providerOverride directly without querying the user table', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      const mockModel = { _generate: jest.fn() } as unknown as BaseChatModel;

      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      cryptoService.decrypt.mockReturnValueOnce('decrypted-key');
      factoryService.createModel.mockReturnValueOnce(mockModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.ANTHROPIC,
      });

      expect(prismaService.user.findUniqueOrThrow).not.toHaveBeenCalled();
      expect(result).toBe(mockModel);
    });

    it('should return createPlatformModel() result for PLATFORM provider', async () => {
      const mockModel = { _generate: jest.fn() } as unknown as BaseChatModel;
      factoryService.createPlatformModel.mockReturnValueOnce(mockModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.PLATFORM,
      });

      expect(factoryService.createPlatformModel).toHaveBeenCalled();
      expect(result).toBe(mockModel);
    });

    it('should throw NotFoundException when no config row exists for user provider', async () => {
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(null);

      await expect(
        llmService.getActiveModel(TEST_USER_UUID, {
          providerOverride: LlmProvider.ANTHROPIC,
        }),
      ).rejects.toThrow(
        new NotFoundException(
          `No configuration found for provider ${LlmProvider.ANTHROPIC}`,
        ),
      );
    });

    it('should throw BadRequestException when config row has no apiKey', async () => {
      const configWithoutApiKey = TestLlmFactory.createPrismaLlmConfigWithoutApiKey();
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(configWithoutApiKey);

      await expect(
        llmService.getActiveModel(TEST_USER_UUID, {
          providerOverride: LlmProvider.ANTHROPIC,
        }),
      ).rejects.toThrow(
        new BadRequestException(
          `Provider ${LlmProvider.ANTHROPIC} is missing an API key`,
        ),
      );
    });

    it('should call crypto.decrypt with the stored ciphertext and pass decrypted key to factory', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig({
        encryptedApiKey: 'CIPHERTEXT',
      });
      const mockModel = { _generate: jest.fn() } as unknown as BaseChatModel;
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      cryptoService.decrypt.mockReturnValueOnce('plain-decrypted-key');
      factoryService.createModel.mockReturnValueOnce(mockModel);

      await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.ANTHROPIC,
      });

      expect(cryptoService.decrypt).toHaveBeenCalledWith('CIPHERTEXT');
      expect(factoryService.createModel).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'plain-decrypted-key' }),
      );
    });

    it('should read activeLlmProvider from user record when no override given', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      const mockModel = { _generate: jest.fn() } as unknown as BaseChatModel;

      prismaService.user.findUniqueOrThrow.mockResolvedValueOnce({
        ...UserFactory.createPrismaUser(),
        activeLlmProvider: LlmProvider.ANTHROPIC,
      });
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      cryptoService.decrypt.mockReturnValueOnce('decrypted-key');
      factoryService.createModel.mockReturnValueOnce(mockModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID);

      expect(prismaService.user.findUniqueOrThrow).toHaveBeenCalledWith({
        where: { id: TEST_USER_UUID },
        select: { activeLlmProvider: true },
      });
      expect(result).toBe(mockModel);
    });

    it('wraps _generate through the limiter with the derived key + limit by default', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      const origGenerate = jest.fn();
      const rawModel = { _generate: origGenerate } as unknown as BaseChatModel;
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      cryptoService.decrypt.mockReturnValueOnce('decrypted-key');
      factoryService.createModel.mockReturnValueOnce(rawModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.ANTHROPIC,
      });

      // _generate is shadowed, and invoking it routes through the limiter pool.
      const generate = (result as unknown as { _generate: (...a: unknown[]) => unknown })
        ._generate;
      expect(generate).not.toBe(origGenerate);
      await generate([], {}, undefined);
      expect(rateLimiterService.schedule).toHaveBeenCalledWith(
        `user:${TEST_USER_UUID}:${LlmProvider.ANTHROPIC}`,
        DEFAULT_MODEL_CONCURRENCY,
        undefined,
        expect.any(Function), // wrapped _generate
        expect.any(Function), // onAdmitted queue-wait telemetry
      );
    });

    it('returns the unwrapped model when rateLimited is false', async () => {
      const mockConfig = TestLlmFactory.createPrismaLlmConfig();
      const origGenerate = jest.fn();
      const rawModel = { _generate: origGenerate } as unknown as BaseChatModel;
      prismaService.llmConfig.findUnique.mockResolvedValueOnce(mockConfig);
      cryptoService.decrypt.mockReturnValueOnce('decrypted-key');
      factoryService.createModel.mockReturnValueOnce(rawModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.ANTHROPIC,
        rateLimited: false,
      });

      expect(result).toBe(rawModel);
      // _generate is untouched (not wrapped).
      expect((result as unknown as { _generate: unknown })._generate).toBe(origGenerate);
    });

    it('always rate-limits PLATFORM even when rateLimited is false', async () => {
      const origGenerate = jest.fn();
      const rawModel = { _generate: origGenerate } as unknown as BaseChatModel;
      factoryService.createPlatformModel.mockReturnValueOnce(rawModel);

      const result = await llmService.getActiveModel(TEST_USER_UUID, {
        providerOverride: LlmProvider.PLATFORM,
        rateLimited: false,
      });

      // Wrapped despite rateLimited:false - _generate was shadowed.
      expect((result as unknown as { _generate: unknown })._generate).not.toBe(
        origGenerate,
      );
    });
  });
});
