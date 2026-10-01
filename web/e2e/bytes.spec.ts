import { expect, test } from "@playwright/test";
import { readRangeResponse } from "../src/bytes";

test("range response accepts an EOF-clipped body", async () => {
  const response = new Response(new Uint8Array([1, 2, 3]), {
    status: 206, headers: { "content-range": "bytes 5-7/8" },
  });
  const out = await readRangeResponse(response, 5, 10);
  expect(out.total).toBe(8);
  expect([...out.bytes]).toEqual([1, 2, 3]);
});

test("a malformed range is cancelled before its body is consumed", async () => {
  let cancelled = false;
  const body = new ReadableStream({ cancel() { cancelled = true; } });
  await expect(readRangeResponse(new Response(body, { status: 200 }), 0, 1024)).rejects.toThrow("require 206");
  expect(cancelled).toBe(true);
});

for (const size of [2, 4]) {
  test(`rejects a ${size}-byte body for a 3-byte range without Content-Length`, async () => {
    const response = new Response(new Uint8Array(size), { status: 206, headers: { "content-range": "bytes 0-2/3" } });
    await expect(readRangeResponse(response, 0, 3)).rejects.toThrow(size < 3 ? "truncated" : "exceeds");
  });
}
