// Synthetic protocol fixtures, not predictions or labelled accuracy evidence.
import { createHash } from 'node:crypto';
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export function scanFixture(points) {
  const raw = Buffer.alloc(points.length * 16);
  points.forEach((p, i) => p.forEach((v, j) => raw.writeFloatLE(v, i * 16 + j * 4)));
  return raw;
}
export function aiFixture(options = {}) {
  const points = options.points ?? [[10,0,0,.5],[20,0,0,.6],[0,10,0,.25],[0,-10,0,.1],
    [0,0,10,.3],[200,0,0,.4],[-10,-0,0,.8],[-10,0,0,.8],[10,-1,0,.15]];
  const raw = scanFixture(points), mask = Buffer.from(options.mask ?? [1,0,0,0,0,0,0,0,0]);
  const retained = Buffer.concat(points.flatMap((_, i) => mask[i] ? [] : [raw.subarray(i*16,(i+1)*16)]));
  const candidates = Buffer.concat(points.flatMap((_, i) => mask[i] ? [raw.subarray(i*16,(i+1)*16)] : []));
  const projection = options.projection ?? {projected:6,occluded:1,invalid:0,outside_view:1,outside_range:1};
  const classes = ['unlabeled','car','bicycle','motorcycle','truck','other-vehicle','person','bicyclist','motorcyclist',
    'road','parking','sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign'];
  const counts = Array(20).fill(0); counts[1] = options.carCount ?? 1; counts[9] = projection.projected-counts[1];
  const report = {experimental:true,model:'SalsaNext SemanticKITTI20',backend:'wgpu',device_selection:'discrete_gpu_0',source_modified:false,
    source_weight_sha256:'1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811',
    onnx_sha256:'8d61577f0d9582adfa28baea4246f915c29d4c5d5e3b1d27adac83ffef4815f5',weights_sha256:'1'.repeat(64),
    class_names:classes,input_sha256:sha(raw),retained_sha256:sha(retained),candidate_sha256:sha(candidates),mask_bytes:points.length,
    sensor_profile:{height:64,width:2048,fov_up_deg:3,fov_down_deg:-25,max_range_m:100,remission:'unchanged normalized0..1'},
    projection:{points:points.length,...projection},selection:{removed:candidates.length/16,retained:retained.length/16,
      unclassified_retained:points.length-projection.projected,threshold:.9,target_classes:['car'],class_counts:counts}};
  const files = () => [new File([raw],'scan.bin'),new File([mask],'candidate-mask.u8'),new File([JSON.stringify(report)],'report.json')];
  return {raw,mask,retained,candidates,report,files};
}
