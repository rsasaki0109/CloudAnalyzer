//! Native full-density COPC query with fixed per-item limits and raw LAS output.
use ca_core::io::{
    copc::CopcHeader,
    copc_query::{CopcQuery, QueryItem, QueryLimits},
};
use std::{
    env,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    time::{Duration, Instant},
};

fn range(file: &mut File, offset: u64, size: u64) -> std::io::Result<Vec<u8>> {
    let mut bytes = vec![
        0;
        usize::try_from(size)
            .map_err(|_| std::io::Error::other("range exceeds address space"))?
    ];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 9 {
        return Err("usage: copc_query INPUT OUT_RAW_OR_- XMIN YMIN ZMIN XMAX YMAX ZMAX".into());
    }
    let bounds: [f64; 6] = args[3..]
        .iter()
        .map(|v| v.parse())
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();
    let mut file = File::open(&args[1])?;
    let file_size = file.metadata()?.len();
    let start = Instant::now();
    let prefix = range(&mut file, 0, file_size.min(1 << 16))?;
    let needed = CopcHeader::needed(&prefix).ok_or("not a LAS/COPC header")?;
    if needed > 8 << 20 {
        return Err("COPC header exceeds 8 MiB".into());
    }
    let head = if needed > prefix.len() {
        range(&mut file, 0, needed as u64)?
    } else {
        prefix[..needed].to_vec()
    };
    let header = CopcHeader::parse(&head)?;
    let limits = QueryLimits::default();
    let mut query = CopcQuery::new(&header, file_size, bounds, limits)?;
    let mut output = if args[2] == "-" {
        None
    } else {
        Some(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args[2])?,
        ))
    };
    let mut selected = 0u64;
    let mut decoded = 0u64;
    let mut read_bytes = prefix.len() as u64
        + if needed > prefix.len() {
            needed as u64
        } else {
            0
        };
    let mut nodes = 0u64;
    let mut pages = 0u64;
    let mut range_time = Duration::ZERO;
    let mut decode_time = Duration::ZERO;
    let mut select_write_time = Duration::ZERO;
    let scale = header.scale();
    let offset = header.offset();
    let record_len = header.record_len();
    while let Some(item) = query.next_item() {
        match item {
            QueryItem::Page { offset, size } => {
                let tick = Instant::now();
                let bytes = range(&mut file, offset, size)?;
                range_time += tick.elapsed();
                read_bytes += size;
                query.supply_page(offset, &bytes)?;
                pages += 1;
            }
            QueryItem::Node(entry) => {
                let tick = Instant::now();
                let chunk = range(&mut file, entry.offset, entry.byte_size as u64)?;
                range_time += tick.elapsed();
                read_bytes += chunk.len() as u64;
                let tick = Instant::now();
                let raw = header.decode_records(
                    &chunk,
                    entry.point_count as usize,
                    limits.raw_node_bytes,
                )?;
                decode_time += tick.elapsed();
                decoded += entry.point_count as u64;
                let tick = Instant::now();
                for record in raw.chunks_exact(record_len) {
                    let point = std::array::from_fn(|a| {
                        i32::from_le_bytes(record[a * 4..a * 4 + 4].try_into().unwrap()) as f64
                            * scale[a]
                            + offset[a]
                    });
                    if query.contains(point) {
                        selected += 1;
                        if let Some(writer) = &mut output {
                            writer.write_all(record)?;
                        }
                    }
                }
                select_write_time += tick.elapsed();
                nodes += 1;
                query.advance_node()?;
            }
        }
    }
    if let Some(writer) = &mut output {
        writer.flush()?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "total_points": header.total_points, "selected_points": selected, "decoded_points": decoded,
            "read_bytes": read_bytes, "file_bytes": file_size, "nodes": nodes, "pages": pages,
            "record_bytes": record_len, "point_format": header.point_format(), "scale": scale, "offset": offset,
            "seconds": start.elapsed().as_secs_f64(), "page_byte_limit": limits.page_bytes,
            "range_seconds": range_time.as_secs_f64(), "decode_seconds": decode_time.as_secs_f64(),
            "select_write_seconds": select_write_time.as_secs_f64(),
            "compressed_node_byte_limit": limits.compressed_node_bytes, "raw_node_byte_limit": limits.raw_node_bytes
        }))?
    );
    Ok(())
}
