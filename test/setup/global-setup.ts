import 'tsconfig-paths/register';

import './resolve-js-extensions';
import { TestSetup } from './test-setup';

export default async function globalSetup() {
  await TestSetup.setupApp();
}
