import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import { BadRequestException } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import { LlmProvider } from '@generated/prisma/client';

import { TEST_USER_UUID, TestLlmFactory } from '@/test/factories';

import type { ProviderParamDto, SaveLlmConfigDto, SetActiveProviderDto } from './dto';
import { LlmController } from './llm.controller';
import { LlmService } from './llm.service';

describe('LlmController', () => {
  let llmController: LlmController;
  let llmService: DeepMocked<LlmService>;

  const userId = TEST_USER_UUID;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [LlmController],
    })
      .useMocker(createMock)
      .compile();

    llmController = module.get(LlmController);
    llmService = module.get(LlmService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(llmController).toBeDefined();
  });

  describe('listProviders', () => {
    it('should delegate to llmService.listProviders with userId', async () => {
      const mockResult = {
        activeProvider: null,
        providers: [],
      };
      llmService.listProviders.mockResolvedValueOnce(mockResult);

      const result = await llmController.listProviders(userId);

      expect(llmService.listProviders).toHaveBeenCalledWith(userId);
      expect(llmService.listProviders).toHaveBeenCalledTimes(1);
      expect(result).toBe(mockResult);
    });
  });

  describe('setActiveProvider', () => {
    it('should pass body.provider to llmService.setActiveProvider', async () => {
      const body = { provider: LlmProvider.ANTHROPIC } as SetActiveProviderDto;
      const mockResult = { activeProvider: LlmProvider.ANTHROPIC };
      llmService.setActiveProvider.mockResolvedValueOnce(mockResult);

      const result = await llmController.setActiveProvider(body, userId);

      expect(llmService.setActiveProvider).toHaveBeenCalledWith(
        userId,
        LlmProvider.ANTHROPIC,
      );
      expect(llmService.setActiveProvider).toHaveBeenCalledTimes(1);
      expect(result).toBe(mockResult);
    });

    it('should pass null provider to llmService.setActiveProvider', async () => {
      const body = { provider: null } as SetActiveProviderDto;
      const mockResult = { activeProvider: null };
      llmService.setActiveProvider.mockResolvedValueOnce(mockResult);

      await llmController.setActiveProvider(body, userId);

      expect(llmService.setActiveProvider).toHaveBeenCalledWith(userId, null);
      expect(llmService.setActiveProvider).toHaveBeenCalledTimes(1);
    });
  });

  describe('getProviderConfig', () => {
    it('should delegate to llmService.getProviderConfig with params.provider', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      const mockResult = TestLlmFactory.createLlmConfigResponse();
      llmService.getProviderConfig.mockResolvedValueOnce(mockResult);

      const result = await llmController.getProviderConfig(params, userId);

      expect(llmService.getProviderConfig).toHaveBeenCalledWith(
        userId,
        LlmProvider.ANTHROPIC,
      );
      expect(llmService.getProviderConfig).toHaveBeenCalledTimes(1);
      expect(result).toBe(mockResult);
    });
  });

  describe('saveProviderConfig', () => {
    it('should throw BadRequestException when body provider does not match URL provider', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      const body = {
        provider: LlmProvider.GOOGLE_GEMINI,
        model: 'gemini-2.0-flash',
      } as SaveLlmConfigDto;

      await expect(
        llmController.saveProviderConfig(params, body, userId),
      ).rejects.toThrow(
        new BadRequestException('Body provider does not match URL provider'),
      );
    });

    it('should not call service when providers mismatch', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      const body = {
        provider: LlmProvider.GOOGLE_GEMINI,
        model: 'gemini-2.0-flash',
      } as SaveLlmConfigDto;

      await expect(
        llmController.saveProviderConfig(params, body, userId),
      ).rejects.toThrow();

      expect(llmService.saveProviderConfig).not.toHaveBeenCalled();
    });

    it('should delegate to service when providers match', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      const body = {
        provider: LlmProvider.ANTHROPIC,
        model: 'claude-3-5-haiku-20241022',
        apiKey: 'test-key',
      } as SaveLlmConfigDto;
      const mockResult = TestLlmFactory.createLlmConfigResponse();
      llmService.saveProviderConfig.mockResolvedValueOnce(mockResult);

      const result = await llmController.saveProviderConfig(params, body, userId);

      expect(llmService.saveProviderConfig).toHaveBeenCalledWith(userId, body);
      expect(llmService.saveProviderConfig).toHaveBeenCalledTimes(1);
      expect(result).toBe(mockResult);
    });
  });

  describe('deleteProviderConfig', () => {
    it('should delegate to llmService.deleteProviderConfig and return void', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      llmService.deleteProviderConfig.mockResolvedValueOnce(undefined);

      const result = await llmController.deleteProviderConfig(params, userId);

      expect(llmService.deleteProviderConfig).toHaveBeenCalledWith(
        userId,
        LlmProvider.ANTHROPIC,
      );
      expect(llmService.deleteProviderConfig).toHaveBeenCalledTimes(1);
      expect(result).toBeUndefined();
    });
  });

  describe('testProviderConfig', () => {
    it('should delegate to llmService.testProviderConfig and return result', async () => {
      const params = {
        provider: LlmProvider.ANTHROPIC,
      } as ProviderParamDto;
      const mockResult = {
        provider: LlmProvider.ANTHROPIC,
        success: false,
        message: 'Auth failed',
        responseTime: 123,
      };
      llmService.testProviderConfig.mockResolvedValueOnce(mockResult);

      const result = await llmController.testProviderConfig(params, userId);

      expect(llmService.testProviderConfig).toHaveBeenCalledWith(
        userId,
        LlmProvider.ANTHROPIC,
      );
      expect(llmService.testProviderConfig).toHaveBeenCalledTimes(1);
      expect(result).toBe(mockResult);
    });
  });
});
