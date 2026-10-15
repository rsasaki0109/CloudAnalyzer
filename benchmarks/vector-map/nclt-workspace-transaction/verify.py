"""Verify local browser evidence against the original immutable workspace ZIP."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
receipt = json.loads((root / "transaction-receipt.json").read_text())
assert all(receipt[k] is True for k in ("cloudExportsExact", "poseGraphExact", "hdMapExact"))
assert receipt["zipSha256"] == "c918ad9dfd2a9972ee385d9e9cf2c2956a3a7cf2fc5d4ece348b7a65177b7db4"
archive = Path(sys.argv[1]).read_bytes()
assert len(archive) == receipt["zipBytes"] == 54815611
assert hashlib.sha256(archive).hexdigest() == receipt["zipSha256"]
with zipfile.ZipFile(sys.argv[1]) as z:
    manifest = json.loads(z.read("manifest.json"))
    for member in manifest["files"]:
        data = z.read(member["path"])
        assert len(data) == member["bytes"], member["path"]
        assert hashlib.sha256(data).hexdigest() == member["sha256"], member["path"]
    project = json.loads(z.read("project.json"))
    assert len(project["session"]["clouds"]) == len(receipt["hashes"]) == 2
    members = {m["name"]: m for m in manifest["files"]}
    for cloud, output in zip(project["session"]["clouds"], receipt["hashes"]):
        expected = members[cloud["source"]["name"]]
        assert output["name"] == cloud["name"]
        assert output["bytes"] == expected["bytes"]
        assert output["sha256"] == expected["sha256"]
print("Archive, member hashes and exact-export receipt verified.")
