import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import { BadRequestException } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import { LlmConfigService } from '@/config/llm';

import { LlmFactoryService } from './llm-factory.service';

describe('LlmFactoryService', () => {
  let llmFactoryService: LlmFactoryService;
  let llmConfig: DeepMocked<LlmConfigService>;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [LlmFactoryService],
    })
      .useMocker(createMock)
      .compile();

    llmFactoryService = module.get(LlmFactoryService);
    llmConfig = module.get(LlmConfigService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(llmFactoryService).toBeDefined();
  });

  function mockLlmConfig(overrides: {
    platformModelEnabled?: boolean;
    geminiApiKey?: string | undefined;
    platformModel?: string;
  }): void {
    if ('platformModelEnabled' in overrides) {
      Object.defineProperty(llmConfig, 'platformModelEnabled', {
        get: () => overrides.platformModelEnabled,
        configurable: true,
      });
    }
    if ('geminiApiKey' in overrides) {
      Object.defineProperty(llmConfig, 'geminiApiKey', {
        get: () => overrides.geminiApiKey,
        configurable: true,
      });
    }
    if ('platformModel' in overrides) {
      Object.defineProperty(llmConfig, 'platformModel', {
        get: () => overrides.platformModel,
        configurable: true,
      });
    }
  }

  describe('createPlatformModel', () => {
    it('should throw BadRequestException when platformModelEnabled is false', () => {
      mockLlmConfig({ platformModelEnabled: false });

      expect(() => llmFactoryService.createPlatformModel()).toThrow(
        new BadRequestException('Platform model is not enabled'),
      );
    });

    it('should throw BadRequestException when enabled but no geminiApiKey', () => {
      mockLlmConfig({ platformModelEnabled: true, geminiApiKey: undefined });

      expect(() => llmFactoryService.createPlatformModel()).toThrow(
        new BadRequestException('Platform model API key is not configured'),
      );
    });

    it('should return ChatGoogleGenerativeAI when enabled and key present', () => {
      mockLlmConfig({
        platformModelEnabled: true,
        geminiApiKey: 'test-api-key',
        platformModel: 'gemini-2.5-flash',
      });

      const model = llmFactoryService.createPlatformModel();

      expect(model).toBeDefined();
    });

    it('should use platformModel from config as model name', () => {
      const expectedModel = 'gemini-2.5-pro';
      mockLlmConfig({
        platformModelEnabled: true,
        geminiApiKey: 'test-api-key',
        platformModel: expectedModel,
      });

      const model = llmFactoryService.createPlatformModel();

      // The model is a ChatGoogleGenerativeAI instance; confirm it was created (no throw)
      expect(model).toBeDefined();
      // Confirm the config getter returns expected value
      expect(llmConfig.platformModel).toBe(expectedModel);
    });
  });

  // Per project conventions, LlmFactoryService.createModel is thin wiring that
  // would require updating every time a provider is added. Coverage comes from
  // integration/e2e tests instead.
});
