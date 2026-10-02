import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const input=fileURLToPath(new URL('../media-frames/vector-map-features/concat.txt',import.meta.url));
const output=fileURLToPath(new URL('../../docs/images/web/vector-map-features.gif',import.meta.url));
if(!process.env.VECTOR_MAP_PLANNING_DIR || !process.env.VECTOR_MAP_JUNCTION_DIR) throw new Error('Set external planning and junction-ablation directories');
if(!existsSync(input))throw new Error('Capture point features first');
execFileSync('ffmpeg',['-loglevel','error','-y','-f','concat','-safe','0','-i',input,
'-vf','crop=1000:660:280:74,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle','-loop','0',output],{stdio:'inherit'});
console.log(`Wrote ${output}`);
