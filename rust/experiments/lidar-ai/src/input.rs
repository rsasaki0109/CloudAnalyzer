//! Bounded reads shared by native inference and frozen-result completion.
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

/// Check the opened file, then cap the actual read as well: a file can grow
/// after metadata was checked. No unbounded allocation based on its contents.
pub fn read(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > limit as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "input is not a regular file within the byte budget: {}",
                path.display()
            ),
        ));
    }
    read_limited(file, limit)
}

fn read_limited(reader: impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let cap = (limit as u64)
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid input budget"))?;
    let mut bytes = Vec::new();
    reader.take(cap).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input grew beyond the byte budget",
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn accepts_exact_budget_and_preserves_raw_bytes() {
        let bytes = [0, 255, 0x80, 0x7f];
        assert_eq!(read_limited(bytes.as_slice(), bytes.len()).unwrap(), bytes);
        assert!(read_limited(io::empty(), 0).unwrap().is_empty());
    }

    #[test]
    fn growing_reader_stops_at_one_byte_beyond_budget() {
        struct Growing(Rc<Cell<usize>>);
        impl Read for Growing {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                bytes.fill(42);
                self.0.set(self.0.get() + bytes.len());
                Ok(bytes.len())
            }
        }
        let consumed = Rc::new(Cell::new(0));
        let error = read_limited(Growing(consumed.clone()), 1024).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(consumed.get(), 1025);
    }

    #[test]
    fn rejects_non_regular_files_and_oversized_regular_files() {
        assert!(read(Path::new(env!("CARGO_MANIFEST_DIR")), 1024).is_err());
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert!(read(&manifest, 1).is_err());
        assert_eq!(
            read(&manifest, 65536).unwrap(),
            std::fs::read(manifest).unwrap()
        );
    }
}
