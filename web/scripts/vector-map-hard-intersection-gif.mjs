import {execFileSync} from 'node:child_process';
import {existsSync,readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const input=fileURLToPath(new URL('../media-frames/vector-map-hard-intersection/concat.txt',import.meta.url));
const output=fileURLToPath(new URL('../../docs/images/web/vector-map-hard-intersection.gif',import.meta.url));
if(!existsSync(input))throw new Error('Capture the actual hard-intersection UI first');
const verification=JSON.parse(readFileSync(fileURLToPath(new URL('../media-frames/vector-map-hard-intersection/verification.json',import.meta.url)),'utf8'));
if(verification.inputMap!==false||verification.roadLanes!==49||verification.errors.length||!verification.exactEditUndo||!verification.exactAllAdditionUndo||!verification.localOsmRoundtrip||verification.maximumNativeDifference>=1e-7)
  throw new Error('The current 49-lane capture must pass its geometry, Undo and reload checks');
execFileSync('ffmpeg',['-loglevel','error','-y','-f','concat','-safe','0','-i',input,
  '-vf','crop=1000:660:280:74,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
  '-loop','0',output],{stdio:'inherit'});
console.log(`Wrote ${output}`);
