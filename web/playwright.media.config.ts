import { defineConfig } from "@playwright/test";
import base from "./playwright.config";

// README screenshots and GIF frames (`npm run media`), not tests.
export default defineConfig({ ...base, testDir: "media", fullyParallel: false, workers: 1, retries: 0 });
