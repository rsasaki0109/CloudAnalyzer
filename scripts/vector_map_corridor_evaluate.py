"""Post-generation correspondence for explicit one-forward/one-backward drafts.

Select adjacent surveyed lane pairs jointly using BEFORE geometry only. Each
source interval keeps the same three ordered curves and station samples AFTER.
Incomplete, curved, ambiguous or unsupported reference corridors are omitted
and their extent reported. This is a development diagnostic, not lane identity
certification or a geometry-generation input.
"""
from __future__ import annotations
import numpy as np
from vector_map_anchor_evaluate import intervals


def stats(d):
    return {"mean_xy_m": float(d.mean()), "p90_xy_m": float(np.quantile(d, .9)),
            "maximum_xy_m": float(d.max()), "fraction_within_0_5m": float((d <= .5).mean())}


def corridor_comparison(before, after, survey, options, *, include_slots=False):
    a, b = intervals(before), intervals(after)
    common = sorted(set(a) & set(b))
    base = {"role": "Before-selected adjacent opposite-direction survey lanes; three ordered boundary slots map one-to-one to the SAME station samples after. Post-generation only, not certified lane identities.",
            "common_source_path_m": sum(a[k][0] for k in common), "evaluated_path_m": 0.,
            "held_path_m": 0., "assignments": [], "samples": 0, "before": None, "after": None}
    if options.get("forward_lanes", 1) != 1 or options.get("backward_lanes", 1) != 1:
        return {**base, "held_path_m": base["common_source_path_m"], "reason": "only explicit one-forward/one-backward lane pairs are covered"}
    if not common:
        return {**base, "reason": "no common generated source intervals"}
    origin = np.asarray(before["roads"][0].get("operator_reference", before["roads"][0]["reference"])[0][:2])
    end = np.asarray(before["roads"][-1].get("operator_reference", before["roads"][-1]["reference"])[-1][:2])
    heading = (end-origin).astype(float)
    if np.linalg.norm(heading) < .1:
        return {**base, "held_path_m": base["common_source_path_m"], "reason": "source trace has no straight corridor axis"}
    heading /= np.linalg.norm(heading)
    normal = np.array([-heading[1], heading[0]])
    boundaries = {v["id"]: np.asarray(v["geometry"]) for v in survey["boundaries"]}
    def bid(side):
        return side if isinstance(side, int) else side["boundary"]
    def oriented(lane):
        ref = lane["left"]
        line = boundaries[bid(ref)]
        return line[::-1] if isinstance(ref, dict) and ref.get("reversed") else line
    lanes = [lane for lane in survey["lanes"] if lane["kind"] == "driving"]
    candidates = []
    for i, lane in enumerate(lanes):
        ids = {bid(lane[s]) for s in ("left", "right")}
        for other in lanes[i+1:]:
            other_ids = {bid(other[s]) for s in ("left", "right")}
            shared = ids & other_ids
            if len(shared) != 1 or len(ids | other_ids) != 3:
                continue
            directions = []
            for member in (lane, other):
                line = oriented(member)
                delta = line[-1,:2]-line[0,:2]
                directions.append(float(delta @ heading / np.linalg.norm(delta)))
            if min(directions) > -np.cos(np.deg2rad(15)) or max(directions) < np.cos(np.deg2rad(15)):
                continue
            # Entire reference curves must be monotone and close to straight.
            curves = []
            for identity in ids | other_ids:
                line = boundaries[identity]
                station = (line[:,:2]-origin) @ heading
                if station[-1] < station[0]:
                    line = line[::-1]
                    station = station[::-1]
                delta = np.diff(line[:,:2], axis=0)
                lengths = np.linalg.norm(delta, axis=1)
                if np.any(np.diff(station) <= 0) or np.any((delta @ heading)/np.maximum(lengths, 1e-12) < np.cos(np.deg2rad(15))):
                    break
                curves.append((identity, station, line))
            if len(curves) == 3:
                candidates.append(((lane["id"], other["id"]), next(iter(shared)), curves, directions, (ids, other_ids)))
    old, new = [], []
    slot_old, slot_new = [[], [], []], [[], [], []]
    for key in common:
        length, p, q = a[key]
        other_length, ap, aq = b[key]
        if p.shape != ap.shape or p.shape[0] != 3 or not np.isclose(length, other_length):
            raise ValueError("corridor source slots or interval length changed")
        t = np.linspace(0, 1, max(1, int(np.ceil(length/.5)))+1)
        old_points = p[:,None,:]+(q-p)[:,None,:]*t[None,:,None]
        new_points = ap[:,None,:]+(aq-ap)[:,None,:]*t[None,:,None]
        source = np.asarray(key).reshape(2,2)
        stations = (source[0]-origin) @ heading + ((source[1]-source[0]) @ heading)*t
        choices = []
        for lane_ids, shared, curves, directions, lane_sides in candidates:
            if any(stations.min() < s[0]-1e-8 or stations.max() > s[-1]+1e-8 for _,s,_ in curves):
                continue
            targets = [(identity, np.column_stack([np.interp(stations,s,line[:,axis]) for axis in (0,1)])) for identity,s,line in curves]
            targets.sort(key=lambda v:float(((v[1]-origin) @ normal).mean()), reverse=True)
            if targets[1][0] != shared:
                continue
            # Physical left/right order must be stable throughout the interval.
            target = np.asarray([v[1] for v in targets])
            if np.any(np.diff(target @ normal, axis=0) >= 0):
                continue
            centers = [np.mean([((points-origin) @ normal).mean() for identity,points in targets if identity in sides]) for sides in lane_sides]
            traffic_order = (centers[0]-centers[1])*(directions[0]-directions[1])
            if (traffic_order > 0) != options.get("left_hand_traffic", True):
                continue
            cost = np.linalg.norm(old_points[:,:,:2]-target, axis=2).mean()
            # Reference coverage gaps must not select a distant parallel road.
            # This is a correspondence gate, not a generator parameter fit.
            if np.linalg.norm(old_points[:,:,:2]-target, axis=2).max() > 2 * options.get("lane_width", 3.5):
                continue
            choices.append((cost, lane_ids, [v[0] for v in targets], target))
        choices.sort(key=lambda v:v[0])
        if not choices or (len(choices)>1 and choices[1][0]-choices[0][0] < .05):
            base["held_path_m"] += length
            continue
        _, lane_ids, boundary_ids, target = choices[0]
        # Same target XYZ projection and assignment for both stages.
        old.append(np.linalg.norm(old_points[:,:,:2]-target,axis=2).ravel())
        new.append(np.linalg.norm(new_points[:,:,:2]-target,axis=2).ravel())
        if include_slots:
            for j in range(3):
                slot_old[j].append(np.linalg.norm(old_points[j,:,:2]-target[j], axis=1))
                slot_new[j].append(np.linalg.norm(new_points[j,:,:2]-target[j], axis=1))
        base["evaluated_path_m"] += length
        base["assignments"].append({"source_interval": list(key), "survey_lanes": list(lane_ids), "survey_boundaries_left_to_right": boundary_ids})
    if old:
        d, e = np.concatenate(old), np.concatenate(new)
        base.update(samples=len(d), before=stats(d), after=stats(e))
        if include_slots:
            base["boundary_slots_left_to_right"] = [
                {"samples": sum(len(v) for v in slot_old[j]),
                 "before": stats(np.concatenate(slot_old[j])),
                 "after": stats(np.concatenate(slot_new[j]))} for j in range(3)]
    return base
