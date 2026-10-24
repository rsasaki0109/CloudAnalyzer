// Synthetic protocol example, never model/accuracy evidence.
import { aiFixture, sha } from './lidar-ai-fixture.mjs';
export function completionFixture(options = {}) {
  const baseline = options.baseline ?? aiFixture({points:[[10,0,0,.5],[10.02,0,0,.5],[10,.05,0,.5],[0,0,10,.5]],
    mask:[1,0,1,0], carCount:2, projection:{projected:2,occluded:1,invalid:0,outside_view:1,outside_range:0}});
  const added = Buffer.from(options.added ?? [0,1,0,0]);
  const completed = Buffer.from(baseline.mask.map((v,i) => v | added[i]));
  const report = {app:'CloudAnalyzer LiDAR surface completion experiment',version:1,source_modified:false,
    policy:{id:'near-surface-two-seeds-v1',radius_m:.15,min_seed_separation_m:.02,seed_score:Math.fround(.99),
      conflict_score:Math.fround(.9),max_neighborhood_owners:256},base:{input_tensor_sha256:'2'.repeat(64),probabilities_sha256:'3'.repeat(64)}};
  const f = {baseline,added,completed,report};
  const select = mask => Buffer.concat([...mask].flatMap((v,i)=>v?[baseline.raw.subarray(i*16,i*16+16)]:[]));
  f.sync = () => {
    f.retained=select(completed.map(v=>1-v));f.candidates=select(completed);f.addedRecords=select(added);
    Object.assign(report,{input_sha256:baseline.report.input_sha256,completed_mask_sha256:sha(completed),
      added_mask_sha256:sha(added),retained_sha256:sha(f.retained),candidate_sha256:sha(f.candidates),
      counts:{original_removed:baseline.report.selection.removed,added:added.reduce((s,v)=>s+v,0),
        total_removed:completed.reduce((s,v)=>s+v,0),retained:f.retained.length/16,
        distance_rejected:0,conflict_rejected:0,crowded_rejected:0,support_rejected:0}});
    Object.assign(report.base,{report_sha256:sha(Buffer.from(JSON.stringify(baseline.report))),candidate_mask_sha256:sha(baseline.mask),
      onnx_sha256:baseline.report.onnx_sha256,weights_sha256:baseline.report.weights_sha256,
      threshold:baseline.report.selection.threshold,target_classes:[...baseline.report.selection.target_classes]});
  };
  f.sync();
  f.files = () => [...baseline.files(),new File([JSON.stringify(report)],'completion-report.json'),
    new File([completed],'completed-mask.u8'),new File([added],'added-mask.u8')];
  return f;
}
