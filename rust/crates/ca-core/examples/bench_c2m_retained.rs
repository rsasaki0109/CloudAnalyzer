//! Measured-source native fresh/retained C2M protocol.
//! Build with --release --features parallel; input XYZ is never resampled.

#[cfg(feature = "parallel")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use ca_core::delaunay::{MaxEdge, delaunay_25d};
    use ca_core::io::{ScalarField, write_mesh_ply, write_ply};
    use serde_json::json;
    use std::io::{BufRead, Write};
    use std::path::Path;
    use std::time::Instant;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(
            "usage: bench_c2m_retained REFERENCE COMPARED NEW_OUTPUT_DIR THREADS fresh|retained"
                .into(),
        );
    }
    let threads = args[3].parse::<usize>()?;
    let retained = match args[4].as_str() {
        "fresh" => false,
        "retained" => true,
        _ => return Err("expected fresh or retained policy".into()),
    };
    if !(1..=64).contains(&threads) {
        return Err("threads must be 1..64".into());
    }
    let reference = ca_core::read(&args[0], &std::fs::read(&args[0])?)?;
    let compared = ca_core::read(&args[1], &std::fs::read(&args[1])?)?;
    for cloud in [&reference, &compared] {
        if !(200_000..=1_000_000).contains(&cloud.len())
            || !cloud.positions.iter().flatten().all(|v| v.is_finite())
        {
            return Err("expected a full finite measured-source input".into());
        }
    }
    let meshed = delaunay_25d(&reference.positions, MaxEdge::Auto).ok_or("cannot mesh input")?;
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?; // Refuse to overwrite a prior run directory.
    std::fs::write(output.join("geometry.ply"), write_mesh_ply(&meshed.mesh))?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "{}",
        json!({
            "kind":"ready", "threads": pool.current_num_threads(),
            "referencePoints":reference.len(), "queryPoints":compared.len(),
            "vertices":meshed.mesh.vertices.len(), "triangles":meshed.mesh.triangles.len(),
            "maxEdge":meshed.max_edge, "removed":meshed.removed,
            "queryOrder":"original decoded PLY order; unchanged XYZ",
            "chunkSize":4096,
        })
    )?;
    stdout.flush()?;
    let source_geometry_bytes = meshed.mesh.vertices.capacity() * std::mem::size_of::<[f64; 3]>()
        + meshed.mesh.triangles.capacity() * std::mem::size_of::<[u32; 3]>();
    let mut source_mesh = Some(meshed.mesh);
    let mut prepared = None;
    let mut initialized = false;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line == "quit" {
            break;
        }
        if line == "prepare" {
            if initialized {
                return Err("already prepared".into());
            }
            let (elapsed, bytes) = if retained {
                let mesh = source_mesh.take().ok_or("missing original mesh")?;
                let start = Instant::now();
                let value = pool
                    .install(|| ca_core::mesh::PreparedMesh::new(mesh))
                    .ok_or("empty mesh")?;
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                let bytes = value.retained_bytes();
                prepared = Some(value);
                (elapsed, bytes)
            } else {
                (0.0, source_geometry_bytes)
            };
            initialized = true;
            writeln!(
                stdout,
                "{}",
                json!({"kind":"preparation", "policy":args[4], "elapsedMillis":elapsed, "sourceGeometryBytes":source_geometry_bytes, "retainedBytes":bytes, "addedBvhBytes":bytes-source_geometry_bytes})
            )?;
            stdout.flush()?;
            continue;
        }
        if !initialized {
            return Err("prepare before querying".into());
        }
        let signed = match line.as_str() {
            "unsigned" => false,
            "signed" => true,
            _ => return Err("expected prepare, unsigned, signed or quit".into()),
        };
        let start = Instant::now();
        let distances = pool.install(|| {
            if let Some(prepared) = prepared.as_ref() {
                prepared.distances_par(&compared.positions, signed)
            } else {
                ca_core::distance::cloud_to_mesh_par(
                    &compared.positions,
                    source_mesh.as_ref().unwrap(),
                    signed,
                )
                .unwrap()
            }
        });
        let elapsed_millis = start.elapsed().as_secs_f64() * 1e3;
        if distances.len() != compared.len() || !distances.iter().all(|v| v.is_finite()) {
            return Err("incomplete/nonfinite measured C2M result".into());
        }
        let stats = ca_core::DistanceStats::from_distances(&distances).ok_or("empty result")?;
        let statistics = [
            stats.min,
            stats.max,
            stats.mean,
            stats.rms,
            stats.std_dev,
            stats.median,
        ]
        .map(|v| format!("{:016x}", v.to_bits()));
        let raw: Vec<u8> = distances.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(output.join("distances-f64.bin"), raw)?;
        let scalar: Vec<f32> = distances.iter().map(|&v| v as f32).collect();
        std::fs::write(
            output.join("distances.ply"),
            write_ply(
                &compared,
                &[ScalarField {
                    name: "C2M",
                    values: &scalar,
                }],
            )?,
        )?;
        writeln!(
            stdout,
            "{}",
            json!({"kind":"result","elapsedMillis":elapsed_millis,
            "signed":signed,"count":stats.count,"statisticsBits":statistics})
        )?;
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("Build bench_c2m_retained with --features parallel");
    std::process::exit(2);
}
