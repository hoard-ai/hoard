import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Post,
  Query,
  Res,
  UseGuards,
} from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import type { Response } from 'express';

import { GetUser } from '@/auth/decorator';
import { LoggedInGuard } from '@/auth/guard';
import type { Uuid } from '@/common/schemas';

import { ChatRequestDto, GetMessagesQueryDto } from './dto';
import { MessagesService } from './messages.service';

@ApiTags('Messages')
@Controller('messages')
@UseGuards(LoggedInGuard)
export class MessagesController {
  // private readonly logger = new Logger(MessagesController.name);

  constructor(private readonly messagesService: MessagesService) {}

  @Get('history')
  @ApiOperation({ summary: 'Get message history' })
  getMessages(@GetUser('id') userId: Uuid, @Query() query: GetMessagesQueryDto) {
    return this.messagesService.getMessages(query, userId);
  }

  @Post()
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Send a message to the agent' })
  sendMessage(@Body() dto: ChatRequestDto, @GetUser('id') userId: Uuid) {
    return this.messagesService.sendMessage(dto, userId);
  }

  @Post('stream')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Send a message and stream the response' })
  sendMessageStream(
    @Body() dto: ChatRequestDto,
    @Res() res: Response,
    @GetUser('id') userId: Uuid,
  ) {
    // TODO: Implement
    return { dto, res, userId };
  }
}
