import {
  Body,
  Controller,
  HttpCode,
  HttpStatus,
  InternalServerErrorException,
  Post,
  Req,
  UseGuards,
} from '@nestjs/common';
import { ApiBody, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Request } from 'express';
import { ZodResponse } from 'nestjs-zod';

import type { UserWithoutPassword } from '@/user/dto';

import { AuthService } from './auth.service';
import { GetUser } from './decorator';
import { AuthDto, LoginResponseDto, LogoutResponseDto, SignupResponseDto } from './dto';
import { LocalGuard, LoggedInGuard } from './guard';

@ApiTags('Authentication')
@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @Post('signup')
  @ApiOperation({ summary: 'Register a new user' })
  @ZodResponse({
    status: HttpStatus.CREATED,
    description: 'User registered successfully',
    type: SignupResponseDto,
  })
  async registerUser(@Body() dto: AuthDto) {
    const user = await this.authService.registerUser(dto);
    return { message: 'Registration successful', user };
  }

  @HttpCode(HttpStatus.OK)
  @UseGuards(LocalGuard)
  @Post('login')
  @ApiOperation({ summary: 'Login with email and password' })
  @ApiBody({ type: AuthDto })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'User logged in successfully',
    type: LoginResponseDto,
  })
  loginUser(@GetUser() user: UserWithoutPassword) {
    return { message: 'Login successful', user };
  }

  @HttpCode(HttpStatus.OK)
  @UseGuards(LoggedInGuard)
  @Post('logout')
  @ApiOperation({ summary: 'Logout current user' })
  @ZodResponse({
    status: HttpStatus.OK,
    description: 'User logged out successfully',
    type: LogoutResponseDto,
  })
  async logoutUser(@Req() req: Request & { logout: (done: (err: any) => void) => void }) {
    return new Promise<{ message: string }>((resolve, reject) => {
      req.logout((err) => {
        if (err) {
          reject(new InternalServerErrorException('Logout failed'));
        } else {
          resolve({ message: 'Logout successful' });
        }
      });
    });
  }
}
