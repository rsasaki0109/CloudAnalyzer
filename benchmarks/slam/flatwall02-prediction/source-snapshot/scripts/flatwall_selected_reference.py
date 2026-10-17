"""Bounded, explicit Flatwall member selection; syntax is still an assumption."""
from pathlib import Path
import hashlib
import tarfile

import flatwall_trajectory_reference as scoring

MEMBERS = {f"gt/flatwall_{i:02d}.txt" for i in range(1, 9)}
MAX_BYTES = 8_388_608


def read_selected(archive: Path, name: str, *, copy_to=None):
    if name not in MEMBERS:
        raise ValueError("select one literal published Flatwall reference member")
    with tarfile.open(archive, "r:gz") as reader:
        members = reader.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member names")
        matches = [m for m in members if m.name == name]
        if len(matches) != 1 or not matches[0].isfile() or not 0 < matches[0].size <= MAX_BYTES:
            raise ValueError("expected one bounded regular selected reference member")
        with reader.extractfile(matches[0]) as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) != matches[0].size:
            raise ValueError("selected reference size mismatch")
    times, poses, error = scoring.parse_tum_assumption(payload.decode("utf-8"))
    if copy_to is not None:
        with Path(copy_to).open("xb") as stream:
            stream.write(payload)
    return times, poses, dict(member=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        rows=len(times), first_timestamp_ns=int(times[0]), last_timestamp_ns=int(times[-1]),
        maximum_original_quaternion_norm_error=error,
        interpretation="Assumed TUM syntax and reference-world-from-LiDAR; reference-specific axes and clock mapping remain unverified.")
