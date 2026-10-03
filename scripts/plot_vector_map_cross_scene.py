"""Plot source-derived boundaries and a separate surveyed movement regression."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vector_map_quality_audit import digest


def plot(cloud: Path, reference: Path, proof: Path, output: Path) -> None:
    import cloudanalyzer_core as core
    report = json.loads((proof / "evaluation.json").read_text(encoding="utf-8"))
    if digest(cloud) != report["source_sha256"] or digest(reference) != report["reference_sha256"]:
        raise ValueError("plot inputs differ from the audited source/reference")
    for name, expected in report["generation_artifact_sha256"].items():
        if digest(proof / name) != expected:
            raise ValueError("frozen generation changed")
    with cloud.open("rb") as file:
        header = []
        while True:
            line = file.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("DATA "):
                break
        offset = file.tell()
    if "FIELDS x y z rgb" not in header or "DATA binary" not in header:
        raise ValueError("plot expects the original XYZRGB planning PCD")
    count = int(next(line.split()[1] for line in header if line.startswith("POINTS ")))
    points = np.memmap(cloud, offset=offset, dtype=np.float32, mode="r", shape=(count,4))
    surveyed = json.loads(json.loads(core.edit_vector_map_relations(str(reference)))["map_json"])
    name = report["road_generation"]["after"]["final_map"]
    generated = json.loads((proof / "roads" / name).read_text(encoding="utf-8"))
    origin = np.array([3800.,73800.])
    fig, axes = plt.subplots(1,2,figsize=(13,6.3))
    def line(ax, xyz, **kw):
        p = np.asarray(xyz)[:, :2] - origin
        ax.plot(p[:,0],p[:,1],**kw)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("Local X - 3800 (m)")
        ax.set_ylabel("Local Y - 73800 (m)")
        ax.grid(alpha=.12)
    ground = points[(points[:,2]<22)][::25]
    axes[0].scatter(ground[:,0]-origin[0],ground[:,1]-origin[1],s=.4,c="#bec4c8",rasterized=True)
    for b in surveyed["boundaries"]:
        line(axes[0],b["geometry"],color="#99a3ab",lw=.5,alpha=.5)
    for b in generated["boundaries"]:
        line(axes[0],b["geometry"],color="#e39016",lw=1.7)
    extent = np.concatenate([b["geometry"] for b in generated["boundaries"]])[:,:2]-origin
    axes[0].set_xlim(extent[:,0].min()-7,extent[:,0].max()+7)
    axes[0].set_ylim(extent[:,1].min()-7,extent[:,1].max()+7)
    axes[0].set_title("Source-only road generation: 12 fragments\nOrange = generated; grey = later survey comparison",fontsize=10)
    axes[0].text(.02,.02,"0 source flags; 17.5 m deferred\nNearest survey boundary: mean 1.42 m, P90 3.79 m",
                 transform=axes[0].transAxes,fontsize=9,bbox={"facecolor":"white","alpha":.9,"edgecolor":"none"})
    ax = axes[1]
    nearby = points[(points[:,0]>3816)&(points[:,0]<3836)&(points[:,1]>73765)&(points[:,1]<73789)&(points[:,2]<27)][::6]
    ax.scatter(nearby[:,0]-origin[0],nearby[:,1]-origin[1],s=.7,c="#c3c7ca",rasterized=True)
    boundaries = {b["id"]:b for b in surveyed["boundaries"]}
    for lane in surveyed["lanes"]:
        if lane["id"] not in [31,36,34]:
            continue
        for side in ("left","right"):
            ref = lane[side]
            ref = ref if isinstance(ref,int) else ref["boundary"]
            line(ax,boundaries[ref]["geometry"],color="#16886a" if lane["id"]!=34 else "#c25b53",lw=1.3,ls="-" if lane["id"]!=34 else "--")
    for stop_id,color,label in [(348,"#1979b4","348: reference target\noutside 16 m gate"),(351,"#cf4545","351: nearby opposing stop\nBefore: offered; after: held")]:
        s=next(s for s in surveyed["stop_lines"] if s["id"]==stop_id)
        line(ax,s["geometry"],color=color,lw=4)
        xy=np.mean(s["geometry"],axis=0)[:2]-origin
        ax.annotate(label,xy,xytext=(xy[0]-4,xy[1]+(2 if stop_id==351 else -4)),fontsize=9,color=color,
                    bbox={"facecolor":"white","alpha":.9,"edgecolor":"none"},arrowprops={"arrowstyle":"-","color":color})
    head=next(s for s in surveyed["traffic_signals"] if s["id"]==350)
    line(ax,head["geometry"],color="#c29400",lw=5)
    xy=np.mean(head["geometry"],axis=0)[:2]-origin
    ax.annotate("350: vehicle housing",xy,xytext=(xy[0]-5,xy[1]-3),fontsize=9,arrowprops={"arrowstyle":"-"})
    ax.set_xlim(16,36)
    ax.set_ylim(-36,-11)
    ax.set_title("Separate surveyed-context regression: rule 1007\nKeep reviewed lanes 31/36; never replace them with 34",fontsize=10)
    fig.text(.5,.015,"Development-scene audit; no registration fit or semantic accuracy claim. Sample map: Copyright 2020 TIER IV, Inc.",ha="center",fontsize=8,color="#56616c")
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cloud","reference","proof","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    plot(args.cloud,args.reference,args.proof,args.output)
