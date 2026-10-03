"""Plot actual frozen planning drafts, retaining both correspondence diagnostics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from vector_map_anchor_evaluate import reference_document
from vector_map_quality_audit import digest


def plot(source, reference, proof, output):
    evaluation = json.loads((proof / "evaluation.json").read_text(encoding="utf-8"))
    if digest(source) != evaluation["source_sha256"] or digest(reference) != evaluation["reference_sha256"]:
        raise ValueError("plot input differs from frozen evaluation")
    if any(digest(proof/name) != sha for name,sha in evaluation["generation_artifact_sha256"].items()):
        raise ValueError("generation changed after freeze")
    with source.open("rb") as stream:
        header = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("missing PCD header")
            line = raw.decode("ascii").strip()
            header.append(line)
            if line.startswith("DATA "):
                break
        offset = stream.tell()
    if "FIELDS x y z rgb" not in header or "DATA binary" not in header:
        raise ValueError("expected original binary XYZRGB")
    count = int(next(v.split()[1] for v in header if v.startswith("POINTS ")))
    cloud = np.memmap(source,dtype=np.float32,offset=offset,shape=(count,4),mode="r")
    drafts = [json.loads((proof/f"{mode}-2-profiles.json").read_text(encoding="utf-8")) for mode in ("before","after")]
    extent = np.concatenate([np.asarray(b)[:,:2] for draft in drafts for road in draft["roads"] for b in road["boundaries"]])
    lo, hi = extent.min(axis=0)-3, extent.max(axis=0)+3
    origin = lo//10*10
    points = cloud[(cloud[:,0]>lo[0])&(cloud[:,0]<hi[0])&(cloud[:,1]>lo[1])&(cloud[:,1]<hi[1])&(cloud[:,2]<22)][::5]
    survey = reference_document(reference,None)
    used = {s if isinstance(s,int) else s["boundary"] for lane in survey["lanes"] if lane["kind"]=="driving" for s in (lane["left"],lane["right"])}
    fig, axes = plt.subplots(1,3,figsize=(15,6),gridspec_kw={"width_ratios":[1,1,1.1]})
    colors = ["#dd8b20","#137f9e"]
    for i, ax in enumerate(axes[:2]):
        ax.scatter(points[:,0]-origin[0],points[:,1]-origin[1],s=.6,c="#c4c9cd",rasterized=True)
        for boundary in survey["boundaries"]:
            if boundary["id"] in used:
                line = np.asarray(boundary["geometry"])[:,:2]-origin
                ax.plot(*line.T,c="#78858c",lw=.8,alpha=.7)
        for road in drafts[i]["roads"]:
            for boundary in road["boundaries"]:
                ax.plot(*(np.asarray(boundary)[:,:2]-origin).T,c=colors[i],lw=2)
            line = np.asarray(road["operator_reference"])[:,:2]-origin
            ax.plot(*line.T,c="#925fa2",ls="--",lw=1)
            if i:
                ax.plot(*(np.asarray(road["reference"])[:,:2]-origin).T,c=colors[i],ls=":",lw=1)
        ax.set(xlim=(lo[0]-origin[0],hi[0]-origin[0]),ylim=(lo[1]-origin[1],hi[1]-origin[1]),
               xlabel=f"X - {origin[0]:.0f} (m)",ylabel=f"Y - {origin[1]:.0f} (m)",
               title=["Before: source-supported,\nwrong lane placement","After: paired-curb translation,\nunchanged lane priors"][i])
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
        ax.text(.03,.02,"31.765 m retained / 0 m deferred",transform=ax.transAxes,fontsize=9,bbox={"facecolor":"white","edgecolor":"none"})
    measures = [evaluation["fixed_reference_corridors"][2], evaluation["paired_source_intervals"][2]]
    for i, mode in enumerate(("before","after")):
        values = [m[mode]["mean_xy_m"] for m in measures]
        rows = np.array([1.,0.])+(.16 if i==0 else -.16)
        axes[2].barh(rows,values,height=.28,color=colors[i],label=mode.capitalize())
        for y,v in zip(rows,values):
            axes[2].text(v+.03,y,f"{v:.3f} m",va="center",fontsize=9)
    axes[2].set(yticks=[1,0],yticklabels=["Ordered lane-pair slots\n29.765 m / 2 m held", "Fixed before-nearest samples\n31.765 m / no lane pairing"],
                xlim=(0,3.45),ylim=(-.55,1.55),xlabel="Mean XY distance to fixed reference samples (m)",
                title="Ordered slots vs. before-nearest samples\nSame BEFORE targets retained after")
    axes[2].grid(axis="x",alpha=.15)
    axes[2].legend(loc="upper right",fontsize=9)
    fig.suptitle("Source curb pairs correct one straight trace; unconfirmed roads stay unchanged",fontsize=13)
    fig.text(.5,.035,"Actual generated boundaries on original points. Grey survey is posthoc only. Purple: original trace; blue dotted: corrected trace. No teacher geometry.",ha="center",fontsize=8)
    fig.text(.5,.012,"Known development scene; not held-out accuracy. Another road retains a 5.674 m residual. Planning sample: Copyright 2020 TIER IV, Inc.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.94))
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","reference","proof","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    plot(args.source,args.reference,args.proof,args.output)
