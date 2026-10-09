/** A small synthetic delivered pair, with real ZIP directory/CRC/deflate records. */
import { createHash } from 'node:crypto';
import { deflateRawSync } from 'node:zlib';

export const map = {format:'vectormap-ir',version:1,
  lanes:[{id:7,kind:'driving',left:1,right:2}],
  boundaries:[{id:1,kind:{type:'virtual'},geometry:[[0,1.75,2],[10,1.75,2]]},
    {id:2,kind:{type:'virtual'},geometry:[[0,-1.75,2],[10,-1.75,2]]}]};
export const cloud = Buffer.from('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 2\n10 0 2\n0 10 2\n10 10 2\n');
const hash = data => createHash('sha256').update(data).digest('hex');
export function quality(failed = true) {
  const support = {fraction:1,start_supported:true,end_supported:true,insufficient_returns:0,height_mismatches:0};
  return {lanes:[{lane:7,center:support,left:{...support,fraction:failed?0.8:1,height_mismatches:failed?1:0},right:support,needs_review:failed}],
    low_support_lanes:failed?[7]:[],omitted_lanes:[],malformed_lanes:[],limited:false,warnings:[],problems_limited:false,
    problems:failed?[{lane:7,curve:'left',reason:'height_mismatch',from_m:5,to_m:5,points:[[5,1.75,2]]}]:[]};
}
export function fixture() {
  const evidence = {editable:{quality:quality()},reopened_osm:{quality:quality()},
    ground_consensus:{editable:{quality:quality(false)},reopened_osm:{quality:quality(false)}}};
  const data = {map:cloud,graph:Buffer.from('graph'),trajectory:Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n'),
    hd_map:Buffer.from('<osm/>'),hd_editable_map:Buffer.from(JSON.stringify(map)),hd_projector:Buffer.from('projector_type: Local'),
    hd_source_audits:Buffer.from(JSON.stringify(evidence)),layout_hypothesis:Buffer.from('{}'),source_proposal:Buffer.from('{}'),decision_history:Buffer.from('{}')};
  const roles = {}, files = [], entries = [], artifacts = {};
  for (const [role, value] of Object.entries(data)) {
    const path = `files/${role}${role === 'map'?'.ply':'.json'}`;
    roles[role] = path; const descriptor = {path,bytes:value.length,sha256:hash(value)}; files.push(descriptor); entries.push([path,value]);
    if (['map','graph','trajectory','hd_map','hd_editable_map','hd_projector','hd_source_audits'].includes(role)) artifacts[role] = descriptor;
  }
  const manifest = {schema:'cloudanalyzer.mapping_review_bundle.v1',files,roles,attribution:'Synthetic test data; no surveyed accuracy',
    review:{status:'draft_needs_review',candidate_id:2,artifacts,diagnosis:{extent:{generated_length_m:10,trajectory_length_m:20,passes_requested_extent:false}}}};
  return {manifest,entries,evidence};
}
function crc(data) {
  let c = 0xffffffff;
  for (const byte of data) { c ^= byte; for (let i=0;i<8;i++) c = (c>>>1) ^ ((c&1)?0xedb88320:0); }
  return (c ^ 0xffffffff) >>> 0;
}
export function zip(entries, {method=8,zip64Local=true,flags=0,mode=0o100600,declaredSize}={}) {
  const locals=[], central=[]; let offset=0;
  for (const [name, raw] of entries) {
    const value=Buffer.from(raw), text=Buffer.from(name), compressed=method===8?deflateRawSync(value):value;
    const header=Buffer.alloc(30), extra=Buffer.alloc(zip64Local?20:0), dir=Buffer.alloc(46);
    header.writeUInt32LE(0x04034b50,0); header.writeUInt16LE(zip64Local?45:20,4); header.writeUInt16LE(flags,6);
    header.writeUInt16LE(method,8); header.writeUInt32LE(crc(value),14);
    header.writeUInt32LE(zip64Local?0xffffffff:compressed.length,18); header.writeUInt32LE(zip64Local?0xffffffff:value.length,22);
    header.writeUInt16LE(text.length,26); header.writeUInt16LE(extra.length,28);
    if(zip64Local){extra.writeUInt16LE(1,0);extra.writeUInt16LE(16,2);extra.writeBigUInt64LE(BigInt(value.length),4);extra.writeBigUInt64LE(BigInt(compressed.length),12);}
    dir.writeUInt32LE(0x02014b50,0);dir.writeUInt16LE(3<<8|45,4);dir.writeUInt16LE(45,6);dir.writeUInt16LE(flags,8);dir.writeUInt16LE(method,10);
    dir.writeUInt32LE(crc(value),16);dir.writeUInt32LE(compressed.length,20);dir.writeUInt32LE(declaredSize??value.length,24);
    dir.writeUInt16LE(text.length,28);dir.writeUInt32LE((mode<<16)>>>0,38);dir.writeUInt32LE(offset,42);
    const local=Buffer.concat([header,text,extra,compressed]);locals.push(local);central.push(Buffer.concat([dir,text]));offset+=local.length;
  }
  const directory=Buffer.concat(central), end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50,0);end.writeUInt16LE(entries.length,8);end.writeUInt16LE(entries.length,10);
  end.writeUInt32LE(directory.length,12);end.writeUInt32LE(offset,16);
  return Buffer.concat([...locals,directory,end]);
}
export function packageFixture(change = () => {}, options) {
  const data=fixture(); change(data);
  return zip([['manifest.json',Buffer.from(JSON.stringify(data.manifest))],...data.entries],options);
}

export function previewFixture() {
  const data = fixture(), {manifest, entries} = data;
  const header = Buffer.from('ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nproperty float intensity\nend_header\n');
  const body = Buffer.alloc(56);
  for (let i=0;i<2;i++) {body.writeDoubleLE(i*10,i*28);body.writeDoubleLE(0,i*28+8);body.writeDoubleLE(2,i*28+16);body.writeFloatLE(0.1234567,i*28+24);}
  const value=Buffer.concat([header,body]), name='files/display-preview.ply';
  const packed={path:name,bytes:value.length,sha256:hash(value)};
  manifest.files=manifest.files.filter(f=>f.path!==manifest.roles.map);
  data.entries=entries.filter(([p])=>p!==manifest.roles.map);
  delete manifest.roles.map;
  manifest.roles.preview_map=name;manifest.files.push(packed);data.entries.push([name,value]);
  const source={path:'/external/original.ply',bytes:header.length+4*28,sha256:hash(Buffer.from('full original records'))};
  manifest.review.artifacts.map=source;
  manifest.schema='cloudanalyzer.mapping_review_bundle.v2';
  manifest.pointcloud_summary={map_points:4};
  manifest.preview_pointcloud={purpose:'display_only',source:{...source},file:{...packed},source_count:4,preview_count:2,max_preview_points:2,
    every_nth_record:2,first_record:0,record_size_bytes:28,coordinate_frame_changed:false,coordinate_or_attribute_quantization:false,
    original_record_bytes_preserved:true,source_for_saved_audits:'original_full_point_map',full_point_map_included:false};
  return data;
}
export function packagePreview(change=()=>{}) {
  const data=previewFixture();change(data);
  return zip([['manifest.json',Buffer.from(JSON.stringify(data.manifest))],...data.entries]);
}
