import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpStatus,
  Logger,
  Param,
  Post,
  Put,
  Query,
  Session,
  UseGuards,
} from '@nestjs/common';
import { ApiNoContentResponse, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Session as ExpressSession } from 'express-session';
import { ZodResponse } from 'nestjs-zod';

import { AuthService } from '@/auth/auth.service';
import { GetUser } from '@/auth/decorator';
import { AdminGuard, LoggedInGuard } from '@/auth/guard';

import {
  CreateUserDto,
  DeleteSelfDto,
  GetUserParamsDto,
  GetUsersQueryDto,
  PaginatedUsersDto,
  UpdateSelfDto,
  UpdateUserDto,
  UserDto,
  type UserWithoutPassword,
  UserWithoutPasswordDto,
} from './dto';
import { UserService } from './user.service';

@ApiTags('Users')
@UseGuards(LoggedInGuard)
@Controller('users')
export class UserController {
  private readonly logger = new Logger(UserController.name);

  constructor(
    private readonly userService: UserService,
    private readonly authService: AuthService,
  ) {}

  @Get('me')
  @ApiOperation({ summary: 'Get the current user' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Current user',
    type: UserWithoutPasswordDto,
  })
  getMe(@GetUser() user: UserWithoutPassword) {
    return user;
  }

  @Put('me')
  @ApiOperation({ summary: 'Update the current user' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Current user after update',
    type: UserWithoutPasswordDto,
  })
  updateMe(@GetUser() user: UserWithoutPassword, @Body() dto: UpdateSelfDto) {
    return this.userService.updateSelf(user.id, dto);
  }

  @Delete('me')
  @ApiOperation({ summary: 'Delete the current user' })
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiNoContentResponse({ description: 'User successfully deleted' })
  async deleteMe(
    @GetUser() user: UserWithoutPassword,
    @Body() dto: DeleteSelfDto,
    @Session() session: ExpressSession,
  ) {
    await this.userService.deleteSelf(user.id, dto);

    try {
      await this.authService.destroyUserSession(session);
    } catch (error) {
      this.logger.error('Failed to destroy session after account deletion', error);
    }
  }

  @UseGuards(AdminGuard)
  @Post()
  @ApiOperation({ summary: 'Create a new user (Admin only)' })
  @ZodResponse({
    status: HttpStatus.CREATED,
    description: 'User successfully created',
    type: UserWithoutPasswordDto,
  })
  createUser(@Body() dto: CreateUserDto) {
    return this.userService.createUserWithDto(dto);
  }

  @UseGuards(AdminGuard)
  @Get()
  @ApiOperation({ summary: 'Get all users (Admin only)' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'Fetched requested users',
    type: PaginatedUsersDto,
  })
  getAllUsers(@Query() query: GetUsersQueryDto) {
    return this.userService.getAllUsers(query);
  }

  @UseGuards(AdminGuard)
  @Get(':id')
  @ApiOperation({ summary: 'Get user by ID (Admin only)' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'User found',
    type: UserDto,
  })
  getUserById(@Param() params: GetUserParamsDto) {
    return this.userService.getUserById(params.id);
  }

  @UseGuards(AdminGuard)
  @Put(':id')
  @ApiOperation({ summary: 'Update user by ID (Admin only)' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'User successfully updated',
    type: UserWithoutPasswordDto,
  })
  updateUser(@Param() params: GetUserParamsDto, @Body() dto: UpdateUserDto) {
    return this.userService.updateUser(params.id, dto);
  }

  @UseGuards(AdminGuard)
  @Delete(':id')
  @ApiOperation({ summary: 'Delete user by ID (Admin only)' })
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiNoContentResponse({ description: 'User successfully deleted' })
  deleteUser(@Param() params: GetUserParamsDto) {
    return this.userService.deleteUser(params.id);
  }
}
