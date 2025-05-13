import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        headless: true
      },
    },
  ],
  /* Run local dev server before starting the tests */
  webServer: {
    command: 'npm run dev --host',
    port: 5173,
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
