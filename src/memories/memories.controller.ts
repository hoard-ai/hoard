import {
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

import { GetUser } from '@/auth/decorator';
import { LoggedInGuard } from '@/auth/guard';
import type { Uuid } from '@/common/schemas';

import {
  CreateMemoryBlockDto,
  GetMemoryBlockParamsDto,
  UpdateMemoryBlockDto,
} from './dto';
import { MemoriesService } from './memories.service';

@ApiTags('Memories')
@Controller('memories')
@UseGuards(LoggedInGuard)
export class MemoriesController {
  constructor(private readonly memoriesService: MemoriesService) {}

  @Get()
  @ApiOperation({ summary: 'List all memory blocks for the current user' })
  listMemoryBlocks(@GetUser('id') userId: Uuid) {
    return this.memoriesService.listMemoryBlocks(userId);
  }

  @Get(':blockLabel')
  @ApiOperation({ summary: 'Get a specific memory block by label' })
  getMemoryBlock(@Param() params: GetMemoryBlockParamsDto, @GetUser('id') userId: Uuid) {
    return this.memoriesService.getMemoryBlock(userId, params.blockLabel);
  }

  @Post()
  @ApiOperation({ summary: 'Create a new memory block' })
  createMemoryBlock(@Body() dto: CreateMemoryBlockDto, @GetUser('id') userId: Uuid) {
    return this.memoriesService.createMemoryBlock(userId, dto);
  }

  @Put(':blockLabel')
  @ApiOperation({ summary: 'Update a memory block' })
  updateMemoryBlock(
    @Param() params: GetMemoryBlockParamsDto,
    @Body() dto: UpdateMemoryBlockDto,
    @GetUser('id') userId: Uuid,
  ) {
    return this.memoriesService.updateMemoryBlock(userId, params.blockLabel, dto);
  }

  @Delete(':blockLabel')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete a memory block' })
  @ApiNoContentResponse({ description: 'Memory block successfully deleted' })
  deleteMemoryBlock(
    @Param() params: GetMemoryBlockParamsDto,
    @GetUser('id') userId: Uuid,
  ) {
    this.memoriesService.deleteMemoryBlock(userId, params.blockLabel);
  }
}
