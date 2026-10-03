import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const directory=fileURLToPath(new URL('../media-frames/vector-map-supported-intersection/',import.meta.url));
const output=fileURLToPath(new URL('../../docs/images/web/vector-map-supported-intersection.gif',import.meta.url));
const v=JSON.parse(readFileSync(`${directory}/verification.json`,'utf8'));
const currentInputs=JSON.parse(readFileSync(fileURLToPath(new URL('../media/vector-map-supported-intersection-inputs.json',import.meta.url)),'utf8'));
if(JSON.stringify(v.operatorInputs)!==JSON.stringify(currentInputs))
  throw new Error('Operator selections changed; recapture and review the source before converting');
if(v.inputMap!==false||v.referenceInputs.length||v.roadLanes!==59||v.approachFragments!==38||v.selectedConnections!==21||
   v.crosswalks!==7||v.stopMarkings!==2||v.signalHousings!==4||v.beforeSourceReviewLanes!==9||v.afterSourceReviewLanes!==0||
   v.finalSourceReviewLanes!==0||v.structuralErrors!==0||v.connectivityWarnings!==15||v.autowareWarnings!==18||v.unreviewedSignalStopLinks!==2||!v.exactWarningRoundtrip||Math.abs(v.deferredRoadLengthM-34.25890841752784)>1e-7||
   JSON.stringify(v.geometricTargetAdoptions)!==JSON.stringify(currentInputs.relation_adoptions)||
   JSON.stringify(v.unresolvedSignalRules)!==JSON.stringify(currentInputs.unresolved_signal_rules)||
   !v.nearestWrongOrientationHeld||!v.noAutomaticTargetSelection||!v.exactAssociationUndo||
   !Number.isFinite(v.associatedNativeDifference)||v.associatedNativeDifference>=1e-7||
   v.errors.length||!v.exactEditUndo||!v.exactAllAdditionUndo||!v.localOsmRoundtrip||!v.exactRegulatoryLaneAssociations||
   [v.maximumNativeDifference,v.beforeNativeDifference,v.sourceRoadNativeDifference].some(d=>!Number.isFinite(d)||d>=1e-7))
  throw new Error('Capture must verify actual generation, source review, native geometry, explicit target suggestions/adoption, Undo and reload');
execFileSync('ffmpeg',['-loglevel','error','-y','-f','concat','-safe','0','-i',`${directory}/concat.txt`,
  '-vf','crop=1000:660:280:74,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
  '-loop','0',output],{stdio:'inherit'});
console.log(`Wrote ${output}`);
