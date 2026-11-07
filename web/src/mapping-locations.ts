/** Link recorded source-station intervals only to their exact verified proposal. */
export const PROPOSAL_BYTES = 8 * 1024 * 1024;
type Point = [number, number, number];
interface Section { station: number; left: Point; right: Point }
export interface MappingLocations { digest: string; candidates: Map<number, Section[]>; unavailable: string | null }
const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('Invalid source proposal');
  return v as Record<string, unknown>;
};
const finite = (v: unknown): number => {
  if (typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v)>1e12) throw new Error('Invalid source coordinate');
  return v;
};
const point = (v: unknown): Point => {
  if (!Array.isArray(v) || v.length!==3) throw new Error('Invalid source point');
  return v.map(finite) as Point;
};
export function parseMappingLocations(value: unknown, digest: string): MappingLocations {
  try {
    const p=object(value);
    if (!/^[a-f0-9]{64}$/.test(digest) || p.schema!=='cloudanalyzer.corridor_proposals.v1' || p.coordinate_frame!=='local_slam_metres') throw new Error('Unsupported source proposal or coordinate frame');
    if (!Array.isArray(p.candidates) || p.candidates.length>2048) throw new Error('Source candidate limit exceeded');
    const candidates=new Map<number,Section[]>();let count=0;
    for (const value of p.candidates) {
      const c=object(value), id=c.id;
      if (!Number.isSafeInteger(id) || Number(id)<1 || candidates.has(Number(id)) || !Array.isArray(c.sections) || c.sections.length<2) throw new Error('Invalid source candidate');
      count+=c.sections.length;if(count>100_000)throw new Error('Source section limit exceeded');
      const sections=c.sections.map(value=>{const s=object(value);return {station:finite(s.station_m),left:point(s.left),right:point(s.right)};});
      if(sections.some((s,i)=>s.station<0 || (i>0 && s.station<=sections[i-1].station)) || c.from_m!==sections[0].station || c.to_m!==sections.at(-1)!.station) throw new Error('Incomplete or unordered source stations');
      candidates.set(Number(id),sections);
    }
    return {digest,candidates,unavailable:null};
  } catch(error) {return {digest,candidates:new Map(),unavailable:error instanceof Error?error.message:'Invalid source proposal'};}
}
export interface LocatedInterval { left: Point[]; right: Point[] }
export function checkMappingInterval(source: MappingLocations, digest: string | null, candidate: number, from: number, to: number): void {
  if(source.unavailable)throw new Error(source.unavailable);
  if(!digest || digest!==source.digest)throw new Error('This action references a different or unidentified source proposal');
  const sections=source.candidates.get(candidate);
  if(!sections || !Number.isFinite(from) || !Number.isFinite(to) || from>to || from<sections[0].station || to>sections.at(-1)!.station)throw new Error('The complete interval is not available in this source proposal');
 }
export function locateMappingInterval(source: MappingLocations, digest: string | null, candidate: number, from: number, to: number): LocatedInterval {
  checkMappingInterval(source,digest,candidate,from,to);
  const sections=source.candidates.get(candidate)!;
  const at=(station:number):Section=>{
    const exact=sections.find(s=>s.station===station);if(exact)return exact;
    const end=sections.findIndex(s=>s.station>station),a=sections[end-1],b=sections[end],t=(station-a.station)/(b.station-a.station);
    const mix=(key:'left'|'right')=>a[key].map((v,i)=>v+t*(b[key][i]-v)) as Point;
    return {station,left:mix('left'),right:mix('right')};
  };
  const rows=[at(from),...sections.filter(s=>s.station>from && s.station<to),at(to)];
  return {left:rows.map(s=>s.left),right:rows.map(s=>s.right)};
}
