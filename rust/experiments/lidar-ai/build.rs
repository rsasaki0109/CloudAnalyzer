fn main() {
    #[cfg(feature = "gpu")]
    {
        use sha2::{Digest, Sha256};
        use std::{fs, path::Path};
        println!("cargo:rerun-if-env-changed=CA_LIDAR_ONNX");
        let path = std::env::var("CA_LIDAR_ONNX").expect(
            "Set CA_LIDAR_ONNX to the locally exported SalsaNext ONNX; weights are not bundled",
        );
        let onnx = Path::new(&path);
        assert_eq!(onnx.file_stem().and_then(|n| n.to_str()), Some("salsanext"));
        let metadata_path = onnx.parent().unwrap().join("onnx-export.json");
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&metadata_path).expect("export metadata missing"))
                .unwrap();
        assert_eq!(
            metadata["source_weight_sha256"],
            "1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811",
            "unsupported checkpoint/profile"
        );
        assert_eq!(metadata["input_shape"], serde_json::json!([1, 5, 64, 2048]));
        assert_eq!(
            metadata["output_shape"],
            serde_json::json!([1, 20, 64, 2048])
        );
        let length = fs::metadata(onnx).unwrap().len();
        assert!(length <= 128 * 1024 * 1024, "model exceeds128MiB budget");
        let digest = format!("{:x}", Sha256::digest(fs::read(onnx).unwrap()));
        assert_eq!(
            metadata["onnx_sha256"].as_str(),
            Some(digest.as_str()),
            "ONNX bytes differ from exporter metadata"
        );
        println!("cargo:rerun-if-changed={path}");
        println!("cargo:rerun-if-changed={}", metadata_path.display());
        burn_onnx::ModelGen::new()
            .input(&path)
            .out_dir("model/")
            .run_from_script();
        let packed = Path::new(&std::env::var("OUT_DIR").unwrap()).join("model/salsanext.bpk");
        let packed_hash = format!("{:x}", Sha256::digest(fs::read(packed).unwrap()));
        println!("cargo:rustc-env=CA_LIDAR_PACK_SHA={packed_hash}");
        println!("cargo:rustc-env=CA_LIDAR_ONNX_SHA={digest}");
    }
}
