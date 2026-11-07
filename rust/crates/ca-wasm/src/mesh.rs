//! Cloud-to-mesh distances and mesh preparation.

use super::*;

fn mesh_queries(xyz: &[f64]) -> Result<Cow<'_, [[f64; 3]]>, JsError> {
    if !xyz.len().is_multiple_of(3) {
        return Err(JsError::new("positions length must be a multiple of 3"));
    }
    let points = xyz.as_chunks::<3>().0;
    Ok(if USE_MESH_BORROWED_QUERIES {
        Cow::Borrowed(points)
    } else {
        Cow::Owned(points.to_vec())
    })
}

#[cfg(test)]
mod mesh_query_tests {
    use super::*;

    #[test]
    fn mesh_query_storage_preserves_bits_and_selected_ownership() {
        let xyz = [
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0023),
            f64::INFINITY,
            368_000.125,
            3_955_000.75,
            -1e-160,
        ];
        let points = mesh_queries(&xyz).unwrap();
        assert_eq!(
            matches!(points, Cow::Borrowed(_)),
            USE_MESH_BORROWED_QUERIES
        );
        if USE_MESH_BORROWED_QUERIES {
            assert_eq!(points.as_flattened().as_ptr(), xyz.as_ptr());
        }
        for (a, b) in points.as_flattened().iter().zip(xyz) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(mesh_queries(&[]).unwrap().is_empty());
    }
}

/// A triangle mesh (PLY with faces, OBJ, STL), used as a C2M reference.
#[wasm_bindgen]
pub struct Mesh {
    pub(crate) inner: TriangleMesh,
}

#[wasm_bindgen]
impl Mesh {
    /// Parse a mesh file, or return `undefined` when the file is a point
    /// cloud (e.g. a PLY without faces).
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Option<Mesh>, JsError> {
        Ok(ca_core::read_mesh(name, bytes)?.map(|inner| Mesh { inner }))
    }

    #[wasm_bindgen(getter, js_name = vertexCount)]
    pub fn vertex_count(&self) -> usize {
        self.inner.vertices.len()
    }

    #[wasm_bindgen(getter, js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.inner.triangles.len()
    }

    fn as_cloud(&self) -> PointCloud {
        PointCloud {
            positions: self.inner.vertices.clone(),
            colors: None,
            attributes: Vec::new(),
        }
    }

    /// `[minX, minY, minZ, maxX, maxY, maxZ]`.
    pub fn bounds(&self) -> Vec<f64> {
        self.as_cloud()
            .bounds()
            .map(|b| b.min.into_iter().chain(b.max).collect())
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = suggestedShift)]
    pub fn suggested_shift(&self) -> Vec<f64> {
        self.as_cloud().suggested_shift().to_vec()
    }

    /// Interleaved vertex `xyz` minus `shift`, narrowed to `f32` for rendering.
    pub fn positions(&self, shift: &[f64]) -> Result<Vec<f32>, JsError> {
        let shift: [f64; 3] = shift
            .try_into()
            .map_err(|_| JsError::new("shift must have 3 components"))?;
        let mut out = Vec::with_capacity(self.inner.vertices.len() * 3);
        for p in &self.inner.vertices {
            out.extend([
                (p[0] - shift[0]) as f32,
                (p[1] - shift[1]) as f32,
                (p[2] - shift[2]) as f32,
            ]);
        }
        Ok(out)
    }

    /// Triangle vertex indices, three per triangle.
    pub fn indices(&self) -> Vec<u32> {
        self.inner.triangles.as_flattened().to_vec()
    }

    /// Interleaved vertex `xyz` at full precision, for [`mesh_distances`].
    #[wasm_bindgen(js_name = rawVertices)]
    pub fn raw_vertices(&self) -> Vec<f64> {
        self.inner.vertices.as_flattened().to_vec()
    }

    /// Serialize the mesh as `"ply"` (binary) or `"obj"`.
    pub fn export(&self, format: &str) -> Result<Vec<u8>, JsError> {
        match format {
            "ply" => Ok(ca_core::io::write_mesh_ply(&self.inner)),
            "obj" => Ok(ca_core::io::write_obj(&self.inner)),
            other => Err(JsError::new(&format!("unknown mesh format {other:?}"))),
        }
    }
}

/// Distance from every point of `compared` to `mesh` (C2M); negative behind
/// the closest triangle when `signed`.
#[wasm_bindgen(js_name = cloudToMesh)]
pub fn cloud_to_mesh(
    compared: &Cloud,
    mesh: &Mesh,
    signed: bool,
    sort_median: Option<bool>,
) -> Result<C2cResult, JsError> {
    let distances = ca_core::cloud_to_mesh(&compared.inner.positions, &mesh.inner, signed)
        .ok_or_else(|| JsError::new("mesh has no triangles"))?;
    summarize_distances(&distances, sort_median)
}

/// C2M distances for a batch of interleaved `xyz` queries against a mesh
/// given as raw vertices and triangle indices. Used by the worker pool.
#[wasm_bindgen(js_name = meshDistances)]
pub fn mesh_distances(
    vertices: &[f64],
    indices: &[u32],
    queries: &[f64],
    signed: bool,
) -> Result<Vec<f64>, JsError> {
    if !vertices.len().is_multiple_of(3) || !indices.len().is_multiple_of(3) {
        return Err(JsError::new("vertices and indices must come in triples"));
    }
    let mut mesh = TriangleMesh {
        vertices: vertices.as_chunks::<3>().0.to_vec(),
        triangles: indices.as_chunks::<3>().0.to_vec(),
    };
    mesh.validate();
    let points = mesh_queries(queries)?;
    ca_core::cloud_to_mesh(&points, &mesh, signed)
        .ok_or_else(|| JsError::new("mesh has no triangles"))
}

/// A worker-owned immutable mesh/BVH. Query ordering and guesses reset on
/// every computation, preserving the fresh batch's exact signed/tie semantics.
#[wasm_bindgen]
pub struct MeshPrepared {
    inner: ca_core::mesh::PreparedMesh,
}

#[wasm_bindgen]
impl MeshPrepared {
    #[wasm_bindgen(js_name = gpuBuffers)]
    pub fn gpu_buffers(&mut self) -> MeshGpuBuffers {
        let (origin, nodes, triangles) = self.inner.gpu_buffers();
        MeshGpuBuffers {
            origin: origin.to_vec(),
            nodes,
            triangles,
        }
    }

    #[wasm_bindgen(js_name = distancesHinted)]
    pub fn distances_hinted(
        &self,
        queries: &[f64],
        signed: bool,
        candidates: &[u32],
    ) -> Result<Vec<f64>, JsError> {
        let points = mesh_queries(queries)?;
        self.inner
            .distances_hinted(&points, signed, candidates)
            .ok_or_else(|| JsError::new("invalid mesh candidate count"))
    }
    #[wasm_bindgen(constructor)]
    pub fn new(vertices: &[f64], indices: &[u32]) -> Result<MeshPrepared, JsError> {
        if !vertices.len().is_multiple_of(3) || !indices.len().is_multiple_of(3) {
            return Err(JsError::new("vertices and indices must come in triples"));
        }
        let mesh = TriangleMesh {
            vertices: vertices.as_chunks::<3>().0.to_vec(),
            triangles: indices.as_chunks::<3>().0.to_vec(),
        };
        let inner = ca_core::mesh::PreparedMesh::new(mesh)
            .ok_or_else(|| JsError::new("mesh has no triangles"))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }

    pub fn distances(&self, queries: &[f64], signed: bool) -> Result<Vec<f64>, JsError> {
        let points = mesh_queries(queries)?;
        Ok(self.inner.distances(&points, signed))
    }
}

#[wasm_bindgen]
pub struct MeshGpuBuffers {
    origin: Vec<f64>,
    nodes: Vec<u32>,
    triangles: Vec<f32>,
}

#[wasm_bindgen]
impl MeshGpuBuffers {
    pub fn origin(&self) -> Vec<f64> {
        self.origin.clone()
    }
    #[wasm_bindgen(js_name = takeNodes)]
    pub fn take_nodes(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.nodes)
    }
    #[wasm_bindgen(js_name = takeTriangles)]
    pub fn take_triangles(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.triangles)
    }
}

/// Content-matched query ordering shared by this worker's retained meshes.
#[wasm_bindgen]
pub struct MeshQueryOrder {
    inner: ca_core::mesh::MeshQueryOrder,
}

#[wasm_bindgen]
impl MeshQueryOrder {
    #[wasm_bindgen(constructor)]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: ca_core::mesh::MeshQueryOrder::new(max_bytes),
        }
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[wasm_bindgen(getter)]
    pub fn hit(&self) -> bool {
        self.inner.hit()
    }

    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }

    #[wasm_bindgen(getter, js_name = entryCount)]
    pub fn entry_count(&self) -> usize {
        self.inner.entry_count()
    }

    pub fn distances(
        &mut self,
        mesh: &MeshPrepared,
        queries: &[f64],
        signed: bool,
    ) -> Result<Vec<f64>, JsError> {
        Ok(self
            .inner
            .distances(&mesh.inner, from_flat(queries)?.positions, signed))
    }
}
