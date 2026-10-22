# SalsaNext on native wgpu

This packet records one **actual pretrained SalsaNext inference on a discrete
wgpu GPU**. Source: the pinned public RangeNet example (124,668 points). The
experiment selects car candidates at softmax threshold 0.9; it is not a
semantic-ground-truth evaluation or a whole-car removal guarantee.

| Result | Value |
| --- | ---: |
| Nearest projected source returns | 99,513 |
| Unclassified returns kept | 25,155 |
| Selected car candidates | 3,450 |
| Retained source records | 121,218 |
| CPU/GPU candidate masks | Byte-identical |
| All-pixel argmax agreement | 100% |
| Maximum probability difference | 0.000018715858459472656 |

The independent Python oracle separately projects the source, runs the same ONNX
CNN on CPU and compares probabilities, masks and exact source records. It uses no
semantic labels. CPU/GPU numerical agreement verifies the adapter, not model
accuracy. The 24,855 occluded and 300 out-of-view returns remain untouched; labels
are not spread from front objects into hidden surfaces.

`verification.json` binds the runtime, binary/model/source SHA256, native report,
independent numeric checks, eight invalid-CLI rejections and four Rust unit tests.
The result was frozen before comparison; no teacher/reference map is read.
A separate run of the committed preparation script reproduced the ONNX digest.
Model and raw example provenance are included; neither weights nor scan files,
generated point sets, tensors or images are redistributed here.

[Build, model preparation and reproduction](../../../rust/experiments/lidar-ai/README.md)
describe this optional native package. It does not change the ordinary browser
or `ca` tools. Current input support is a single HDL-64/KITTI sensor scan; a fused
map requires different preprocessing/model validation. A warmed benchmark and
held-out semantic evaluation remain future work. Timing in the report includes
model loading and process-local kernel setup; driver caches were not reset.

Model: [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext),
[architecture attribution](../../../rust/experiments/lidar-ai/third_party/NOTICE.md).
Raw example: [PRBonn/rangenet_lib](https://github.com/PRBonn/rangenet_lib/tree/3fc223ea31251dc4380f6a0013c271d408a5eefa/example).
Upstream model and dataset terms apply independently.
