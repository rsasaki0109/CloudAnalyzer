//! Measured-source native C2M protocol for benchmark-native-mesh-stack.mjs.
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
    if args.len() != 4 {
        return Err("usage: bench_c2m_source REFERENCE COMPARED NEW_OUTPUT_DIR THREADS".into());
    }
    let threads = args[3].parse::<usize>()?;
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
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line == "quit" {
            break;
        }
        let signed = match line.as_str() {
            "unsigned" => false,
            "signed" => true,
            _ => return Err("expected unsigned, signed or quit".into()),
        };
        let start = Instant::now();
        let distances = pool
            .install(|| {
                ca_core::distance::cloud_to_mesh_par(&compared.positions, &meshed.mesh, signed)
            })
            .ok_or("empty mesh")?;
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
    eprintln!("Build bench_c2m_source with --features parallel");
    std::process::exit(2);
}
