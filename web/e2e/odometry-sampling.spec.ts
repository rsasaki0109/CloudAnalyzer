import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("odometry sampling is independent, atomic and retained by worker registrations", async ({ page }) => {
  await page.route("**/__odometry-sampling", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__odometry-sampling.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__odometry-sampling");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(name => /^ca_wasm.*\.wasm$/.test(name));
  const result = await page.evaluate(async wasmPath => {
    const wasm = await import("/__odometry-sampling.js");
    await wasm.default({ module_or_path: wasmPath });
    const points = Array.from({ length: 48 }, (_, k) => {
      const i = Math.floor(k / 4), j = k % 4;
      return [3 * (i % 4) - 6 + 0.03125 + 0.09375 * j, 3 * Math.floor(i / 4) + 2.03125, i % 3 + 0.03125];
    });
    const cloud = wasm.Cloud.parse("sampling.xyz", new TextEncoder().encode(points.map(p => p.join(" ")).join("\n")));
    const odometry = new wasm.LidarOdometry(0, 80, false);
    const worker = new wasm.OdometryMap(odometry.mapVoxel, odometry.mapPoints);
    let legacy: any, dense: any, reset: any;
    try {
      const defaultMissing = odometry.registrationVoxel === undefined;
      odometry.setDiagnostics(true);
      odometry.register(cloud);
      const placed = [...odometry.placed()], before = odometry.lastReport();
      legacy = odometry.begin(cloud);
      const oldSource = [...legacy.source()];
      odometry.setRegistrationVoxel(0.0625);
      dense = odometry.begin(cloud);
      if (JSON.stringify([...dense.source()]) !== JSON.stringify(points.flat())) throw new Error("Source was pre-thinned for insertion");
      let rejected = 0;
      for (const value of [0, -0, -0.1, NaN, Infinity, -Infinity]) {
        try { odometry.setRegistrationVoxel(value); }
        catch (error) {
          if (!String(error).includes("registration_voxel must be finite and positive")) throw error;
          rejected++;
        }
        if (odometry.registrationVoxel !== 0.0625) throw new Error("Rejected value changed sampling");
      }
      if (JSON.stringify([...odometry.placed()]) !== JSON.stringify(placed) || odometry.lastReport() !== before ||
          JSON.stringify([...legacy.source()]) !== JSON.stringify(oldSource)) throw new Error("Configuration changed existing state");
      odometry.setRegistrationVoxel(undefined);
      reset = odometry.begin(cloud);
      if (odometry.registrationVoxel !== undefined || JSON.stringify([...reset.source()]) !== JSON.stringify(oldSource)) throw new Error("Legacy reset changed source");
      worker.update(Float64Array.from(placed), new Float64Array(3), odometry.maxRange);
      const terms = dense.terms(), source = dense.source();
      let wire = 0, finished = false;
      for (let iteration = 0; iteration < 64; iteration++) {
        const equations = worker.equations(source, dense.total(), terms);
        wire = equations.length;
        if (wire !== 42) throw new Error("Worker equation format changed");
        if (odometry.step(dense, equations)) { finished = true; break; }
      }
      if (!finished) throw new Error("Dense worker registration did not finish");
      odometry.finish(dense); dense = undefined;
      const report = JSON.parse(odometry.lastReport());
      if (report.termination !== "converged") throw new Error(report.termination);
      return { defaultMissing, legacy: oldSource.length / 3, dense: source.length / 3,
        insertion: placed.length / 3, reset: reset.source().length / 3, rejected, wire,
        matched: report.correspondences.matched_points, registered: report.registration_points,
        mapVoxel: odometry.mapVoxel, mapPoints: odometry.mapPoints };
    } finally {
      legacy?.free(); dense?.free(); reset?.free(); worker.free(); odometry.free(); cloud.free();
    }
  }, wasmPath);
  expect(result).toEqual({ defaultMissing: true, legacy: 12, dense: 48, insertion: 12,
    reset: 12, rejected: 6, wire: 42, matched: 48, registered: 48, mapVoxel: 1, mapPoints: 20 });
});
