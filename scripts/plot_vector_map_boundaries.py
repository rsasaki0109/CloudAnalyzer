"""Plot measured boundary errors over original points, after generation freeze."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial import cKDTree

from hard_intersection_evaluate import resample
from vector_map_boundary_evaluate import reference_document
from vector_map_quality_audit import digest


def points(path: Path):
    if path.suffix == ".las":
        import laspy
        cloud = laspy.read(path)
        xyz = np.column_stack([cloud.x, cloud.y, cloud.z])
        rgb = np.column_stack([cloud.red, cloud.green, cloud.blue])/65535
        return xyz, rgb
    with path.open("rb") as file:
        header = []
        while True:
            line = file.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("DATA "):
                break
        offset = file.tell()
    if "FIELDS x y z rgb" not in header or "DATA binary" not in header:
        raise ValueError("expected cached XYZRGB binary PCD")
    n = int(next(line.split()[1] for line in header if line.startswith("POINTS ")))
    xyzrgb = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(n,4))
    colors = xyzrgb[:,3].copy().view(np.uint32)
    rgb = np.column_stack([(colors>>16)&255, (colors>>8)&255, colors&255])/255
    return xyzrgb[:,:3], rgb


def plot(cases: list[dict], output: Path):
    fig, axes = plt.subplots(1,len(cases),figsize=(13,6))
    for ax, case in zip(np.atleast_1d(axes), cases):
        proof = Path(case["proof"])
        report = json.loads((proof/"evaluation.json").read_text(encoding="utf-8"))
        source, reference = Path(case["source"]), Path(case["reference"])
        assert digest(source)==report["source_sha256"] and digest(reference)==report["reference_sha256"]
        assert all(digest(proof/name)==sha for name,sha in report["generation_artifact_sha256"].items())
        generated = json.loads((proof/report["cases"]["rgb"][-1]["map"]).read_text(encoding="utf-8"))
        survey = reference_document(reference, case.get("epsg"))
        used = {lane[side] if isinstance(lane[side],int) else lane[side]["boundary"] for lane in survey["lanes"] if lane["kind"]=="driving" for side in ("left","right")}
        lines = [np.asarray(b["geometry"]) for b in survey["boundaries"] if b["id"] in used]
        target = cKDTree(np.concatenate([resample(line,.5) for line in lines])[:,:2])
        boundaries = [np.asarray(b["geometry"]) for b in generated["boundaries"]]
        extent = np.concatenate(boundaries)
        origin = extent[:,:2].mean(axis=0)
        xyz, rgb = points(source)
        selected = (np.abs(xyz[:,2]-np.median(extent[:,2]))<1.0)
        selected &= np.all(xyz[:,:2]>=extent[:,:2].min(axis=0)-6,axis=1)
        selected &= np.all(xyz[:,:2]<=extent[:,:2].max(axis=0)+6,axis=1)
        ids = np.flatnonzero(selected)[::max(1,int(selected.sum()/120000))]
        ax.scatter(xyz[ids,0]-origin[0],xyz[ids,1]-origin[1],c=rgb[ids]*.65,s=.6,rasterized=True,alpha=.7)
        for line in lines:
            ax.plot(line[:,0]-origin[0],line[:,1]-origin[1],color="#617684",lw=.6,alpha=.55)
        segments, errors = [], []
        for line in boundaries:
            sampled = resample(line,.5)
            segments.extend(np.stack([sampled[:-1,:2]-origin,sampled[1:,:2]-origin],axis=1))
            errors.extend(target.query((sampled[:-1,:2]+sampled[1:,:2])/2)[0])
        collection = LineCollection(segments,cmap="plasma",linewidths=2.2)
        collection.set_array(np.asarray(errors))
        collection.set_clim(0,6)
        ax.add_collection(collection)
        m = report["metrics"]["rgb"]
        ax.set_title(f"{case['name']}: source-only generated boundaries\nMean {m['boundary_distance']['mean_xy_m']:.2f} m; P90 {m['boundary_distance']['p90_xy_m']:.2f} m",fontsize=10)
        ax.text(.02,.02,f"Generated {m['generated_path_m']:.1f} m; deferred {m['deferred_path_m']:.1f} m\nRGB sources {m['rgb_source_vertices']}; {m['lane_fragments']} lane fragments",
                transform=ax.transAxes,fontsize=9,bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
        ax.set_aspect("equal")
        ax.set_xlim(extent[:,0].min()-origin[0]-6,extent[:,0].max()-origin[0]+6)
        ax.set_ylim(extent[:,1].min()-origin[1]-6,extent[:,1].max()-origin[1]+6)
        ax.set_xlabel("Source X relative to display centre (m)")
        ax.set_ylabel("Source Y relative to display centre (m)")
    fig.text(.5,.02,"Known development scenes; survey grey, original RGB points darkened for display only. No registration or lane pairing.\nAutoware: Copyright 2020 TIER IV, Inc. / Tokyo: Dynamic Map Platform 2026, CC BY 4.0.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.07,right=.9,bottom=.16,top=.88,wspace=.3)
    fig.colorbar(collection,ax=np.atleast_1d(axes).tolist(),fraction=.02,pad=.02,label="Unpaired nearest surveyed boundary XY distance (m)")
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    plot(json.loads(args.config.read_text(encoding="utf-8")),args.output)
