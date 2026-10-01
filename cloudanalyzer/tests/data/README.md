# Synthetic spatial signal fixture

`signal.copc.laz` contains 5,275 synthetic XYZ points split across the root and
level-one nodes: 275 head-shaped points and 5,000 points outside the test box.
Coordinates use a 0.001 scale; intensity, classification and RGB are artificial.
It is not a measured signal or a source of detection-accuracy evidence. The
minimal writer omits the ordinary LAZ chunk table; read it through a COPC reader.

Regenerate from the repository root:

```sh
cargo run --manifest-path rust/Cargo.toml -p ca-core --example make_test_signal_copc -- cloudanalyzer/tests/data/signal.copc.laz
```
