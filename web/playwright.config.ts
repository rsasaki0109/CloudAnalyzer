import { defineConfig, devices } from "@playwright/test";

// PW_PORT lets several checkouts run their tests side by side.
const port = Number(process.env.PW_PORT) || 4173;

const chromium = {
  ...devices["Desktop Chrome"],
  // Headless Chromium renders WebGL with SwiftShader.
  launchOptions: { executablePath: process.env.PW_CHROMIUM_EXECUTABLE_PATH, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] },
};

// End-to-end tests against the production build (`npm run build` first).
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 30_000 },
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
  },
  projects: [
    // Runs first; when the app cannot start, the other tests are skipped.
    // `--no-deps` leaves it out when running a single spec locally.
    { name: "boot", testMatch: /boot\.setup\.ts$/, use: chromium },
    { name: "chromium", dependencies: ["boot"], use: chromium },
  ],
  webServer: {
    command: `npx vite preview --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: !process.env.CI,
  },
});
