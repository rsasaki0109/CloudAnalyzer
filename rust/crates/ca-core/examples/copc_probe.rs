//! Load a COPC file down to a point budget, as the viewer does:
//! `cargo run --release -p ca-core --example copc_probe -- file.copc.laz [budget]`.

use ca_core::io::copc::{CopcHeader, CopcPoints, NodeSelector};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: copc_probe <file> [budget]")?;
    let budget: u64 = args
        .next()
        .map(|b| b.parse())
        .transpose()?
        .unwrap_or(2_000_000);
    let file = std::fs::read(&path)?;
    let header = CopcHeader::parse(&file[..CopcHeader::needed(&file).ok_or("short")?])?;
    println!(
        "{} points, halfsize {}, spacing {}",
        header.total_points, header.halfsize, header.spacing
    );
    let mut selector = NodeSelector::new(header.root_page);
    let (mut level, mut total) = (0, 0u64);
    loop {
        for (o, s) in selector.pages_for(level) {
            selector.add_page(o, &file[o as usize..(o + s) as usize]);
        }
        let n = selector.level_points(level);
        if n == 0 || total + n > budget {
            break;
        }
        total += n;
        println!("level {level}: {n} points");
        if !selector.deeper_than(level) {
            level += 1;
            break;
        }
        level += 1;
    }
    let nodes = selector.nodes_to(level - 1);
    let bytes: i64 = nodes.iter().map(|e| e.byte_size as i64).sum();
    println!(
        "{} nodes, {:.1} MB compressed",
        nodes.len(),
        bytes as f64 / 1e6
    );
    let start = Instant::now();
    let mut points = CopcPoints::default();
    for e in &nodes {
        let chunk = &file[e.offset as usize..e.offset as usize + e.byte_size as usize];
        points.extend(header.decode_node(chunk, e.point_count as usize)?);
    }
    let cloud = points.into_cloud();
    println!(
        "{} nodes, {} points in {:.0} ms; first {:?}",
        nodes.len(),
        cloud.len(),
        start.elapsed().as_secs_f64() * 1e3,
        cloud.positions.first()
    );
    Ok(())
}
