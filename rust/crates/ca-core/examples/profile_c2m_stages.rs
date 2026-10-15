//! Native C2M stage diagnostics using the unchanged production source modules.
//! Source inclusion keeps profiling helpers out of the library's public API.

pub use ca_core::{PointCloud, kdtree};

#[allow(dead_code)]
#[path = "../src/distance.rs"]
mod distance;
#[allow(dead_code)]
#[path = "../src/mesh.rs"]
mod mesh;

#[cfg(feature = "parallel")]
fn staged(
    points: &[[f64; 3]],
    mesh: &mesh::TriangleMesh,
    query_policy: u8,
) -> (Vec<f64>, [f64; 5]) {
    use rayon::prelude::*;
    use std::time::Instant;
    let millis = |start: Instant| start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let bvh = mesh::MeshBvh::new(mesh).unwrap();
    let build = millis(start);
    let start = Instant::now();
    let order = distance::morton_order(points);
    let morton = millis(start);
    let start = Instant::now();
    let parts: Vec<Vec<f64>> = order
        .par_chunks(4096)
        .map(|chunk| {
            let mut guess = None;
            let mut stack = Vec::new();
            chunk
                .iter()
                .map(|&i| {
                    let hit = match query_policy {
                        3 => bvh.nearest_with_paired_projection(points[i], guess, &mut stack),
                        2 => bvh.nearest_with_cached_bounds_clamped(points[i], guess, &mut stack),
                        1 => bvh.nearest_with_cached_bounds_paired(points[i], guess, &mut stack),
                        _ => bvh.nearest_with_cached_bounds(points[i], guess, &mut stack),
                    };
                    if hit.triangle == usize::MAX {
                        guess = None;
                        f64::INFINITY
                    } else {
                        guess = Some(hit.triangle);
                        hit.distance_sq.sqrt()
                    }
                })
                .collect()
        })
        .collect();
    let query = millis(start);
    let start = Instant::now();
    let mut out = vec![0.0; points.len()];
    for (chunk, values) in order.chunks(4096).zip(parts) {
        for (&i, d) in chunk.iter().zip(values) {
            out[i] = d;
        }
    }
    let scatter = millis(start);
    let start = Instant::now();
    drop(order);
    drop(bvh);
    let destroy = millis(start);
    (out, [build, morton, query, scatter, destroy])
}

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
        return Err("usage: profile_c2m_stages REFERENCE COMPARED NEW_OUTPUT_DIR THREADS".into());
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
            return Err("expected full finite measured-source inputs".into());
        }
    }
    let meshed = delaunay_25d(&reference.positions, MaxEdge::Auto).ok_or("cannot mesh input")?;
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    std::fs::write(output.join("geometry.ply"), write_mesh_ply(&meshed.mesh))?;
    let vertices = meshed.mesh.vertices.len();
    let triangles = meshed.mesh.triangles.len();
    // Move the original vectors: no coordinate copy, transform or reorder.
    let mesh = mesh::TriangleMesh {
        vertices: meshed.mesh.vertices,
        triangles: meshed.mesh.triangles,
    };
    // Read the compiled-in policy once, outside all command clocks. The
    // controller guards the other native policies; no library API is added.
    let source = include_str!("../src/distance.rs");
    let clamped = source.contains("const USE_PAR_MESH_BOUND_PAIR_CLAMP: bool = true;");
    let query_policy =
        if clamped && source.contains("const USE_PAR_MESH_PAIRED_PROJECTION: bool = true;") {
            3
        } else if clamped {
            2
        } else if source.contains("const USE_PAR_MESH_BOUND_PAIR: bool = true;") {
            1
        } else {
            0
        };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "{}",
        json!({"kind":"ready", "threads":pool.current_num_threads(), "chunkSize":4096,
        "referencePoints":reference.len(), "queryPoints":compared.len(), "vertices":vertices,
        "triangles":triangles, "maxEdge":meshed.max_edge, "removed":meshed.removed,
        "queryOrder":"original decoded PLY order; unchanged XYZ"})
    )?;
    stdout.flush()?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line == "quit" {
            break;
        }
        if !matches!(line.as_str(), "whole" | "stages") {
            return Err("expected whole, stages or quit".into());
        }
        let start = Instant::now();
        let (values, stages) = pool.install(|| {
            if line == "stages" {
                let (values, stages) = staged(&compared.positions, &mesh, query_policy);
                (values, Some(stages))
            } else {
                (
                    distance::cloud_to_mesh_par(&compared.positions, &mesh, false).unwrap(),
                    None,
                )
            }
        });
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if values.len() != compared.len() || !values.iter().all(|v| v.is_finite()) {
            return Err("incomplete/nonfinite output".into());
        }
        let stats = distance::DistanceStats::from_distances(&values).ok_or("empty output")?;
        let statistics = [
            stats.min,
            stats.max,
            stats.mean,
            stats.rms,
            stats.std_dev,
            stats.median,
        ]
        .map(|v| format!("{:016x}", v.to_bits()));
        std::fs::write(
            output.join("distances-f64.bin"),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let scalar: Vec<f32> = values.iter().map(|&v| v as f32).collect();
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
            json!({"kind":"result", "mode":line, "signed":false, "elapsedMillis":elapsed,
            "stageMillis":stages, "count":stats.count, "statisticsBits":statistics})
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
    eprintln!("Build profile_c2m_stages with --features parallel");
    std::process::exit(2);
}
