import {
  BadRequestException,
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Post,
  Put,
  UseGuards,
} from '@nestjs/common';
import { ApiNoContentResponse, ApiOperation, ApiTags } from '@nestjs/swagger';
import { ZodResponse } from 'nestjs-zod';

import { GetUser } from '@/auth/decorator';
import { LoggedInGuard } from '@/auth/guard';
import type { Uuid } from '@/common/schemas';

import {
  ActiveProviderResponseDto,
  LlmConfigResponseDto,
  LlmProvidersListDto,
  ProviderParamDto,
  SaveLlmConfigDto,
  SetActiveProviderDto,
  TestConfigResponseDto,
} from './dto';
import { LlmService } from './llm.service';

@ApiTags('LLMs')
@Controller('llms')
@UseGuards(LoggedInGuard)
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Get()
  @ApiOperation({ summary: 'List all LLM providers and their config status' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'All providers with config status',
    type: LlmProvidersListDto,
  })
  async listProviders(@GetUser('id') userId: Uuid) {
    return this.llmService.listProviders(userId);
  }

  @Put('active')
  @ApiOperation({ summary: 'Set the active LLM provider' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Active provider updated',
    type: ActiveProviderResponseDto,
  })
  async setActiveProvider(
    @Body() body: SetActiveProviderDto,
    @GetUser('id') userId: Uuid,
  ) {
    return this.llmService.setActiveProvider(userId, body.provider);
  }

  @Get(':provider')
  @ApiOperation({ summary: 'Get config for a specific provider' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Provider configuration',
    type: LlmConfigResponseDto,
  })
  async getProviderConfig(
    @Param() params: ProviderParamDto,
    @GetUser('id') userId: Uuid,
  ) {
    return this.llmService.getProviderConfig(userId, params.provider);
  }

  @Put(':provider')
  @ApiOperation({ summary: 'Save config for a provider' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Saved provider configuration',
    type: LlmConfigResponseDto,
  })
  async saveProviderConfig(
    @Param() params: ProviderParamDto,
    @Body() body: SaveLlmConfigDto,
    @GetUser('id') userId: Uuid,
  ) {
    if (params.provider !== body.provider) {
      throw new BadRequestException('Body provider does not match URL provider');
    }
    return this.llmService.saveProviderConfig(userId, body);
  }

  @Delete(':provider')
  @ApiOperation({ summary: 'Delete config for a provider' })
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiNoContentResponse({ description: 'Provider config successfully deleted' })
  async deleteProviderConfig(
    @Param() params: ProviderParamDto,
    @GetUser('id') userId: Uuid,
  ) {
    await this.llmService.deleteProviderConfig(userId, params.provider);
  }

  @Post(':provider/test')
  @ApiOperation({ summary: 'Test connectivity for a provider' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Test result',
    type: TestConfigResponseDto,
  })
  async testProviderConfig(
    @Param() params: ProviderParamDto,
    @GetUser('id') userId: Uuid,
  ) {
    return this.llmService.testProviderConfig(userId, params.provider);
  }
}
