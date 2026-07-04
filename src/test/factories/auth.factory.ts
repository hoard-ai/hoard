import type { AuthDto } from '@/auth/dto';

export interface AuthDtoFactoryOptions {
  email?: string;
  password?: string;
}

export class AuthFactory {
  private static defaultAuthDto: AuthDto = {
    email: 'test@example.com',
    password: 'password123',
  };

  static createAuthDto(options: AuthDtoFactoryOptions = {}): AuthDto {
    return {
      ...this.defaultAuthDto,
      ...options,
    };
  }
}
