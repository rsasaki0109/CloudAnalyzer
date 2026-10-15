use ca_core::{PointCloud, distance::PreparedC2cReference};

fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn fixture(count: usize, offset: f64, scale: f64) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            [
                offset + scale * ((i * 19 % 113) as f64 / 17.),
                offset + scale * ((i * 37 % 127) as f64 / 23.),
                scale * ((i * 53 % 89) as f64 / 13.),
            ]
        })
        .collect()
}

#[test]
fn owned_reference_survives_changes_to_source_storage() {
    assert!(PreparedC2cReference::new(&[]).is_none());
    let mut reference = fixture(257, 500000., 1.);
    reference.extend([[500000., 500000., -0.0]; 16]);
    let original = PointCloud {
        positions: reference.clone(),
        colors: None,
        attributes: vec![],
    };
    let prepared = PreparedC2cReference::new(&reference).unwrap();
    let bytes = prepared.retained_bytes();
    reference.fill([1e10; 3]);
    drop(reference);
    for queries in [
        fixture(4097, 500000., 1.),
        fixture(17, 500001., 0.5),
        vec![],
    ] {
        let compared = PointCloud {
            positions: queries.clone(),
            colors: None,
            attributes: vec![],
        };
        let expected = ca_core::distance::cloud_to_cloud(&compared, &original).unwrap();
        assert_eq!(bits(prepared.distances(&queries)), bits(expected));
        assert_eq!(prepared.retained_bytes(), bytes);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn retained_parallel_queries_preserve_serial_and_fresh_results() {
    let pools: Vec<_> = [1, 2, 4]
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .into();
    for (offset, scale) in [(0., 1.), (500000., 1.), (0., 1e-160), (0., 1e150)] {
        let mut reference = fixture(257, offset, scale);
        reference.extend([reference[0]; 16]);
        let prepared = PreparedC2cReference::new(&reference).unwrap();
        let bytes = prepared.retained_bytes();
        for count in [0, 1, 4095, 4096, 4097, 65535, 65536, 65537] {
            let mut queries = fixture(count, offset + scale * 0.125, scale);
            for reversed in [false, true] {
                if reversed {
                    queries.reverse();
                }
                let serial = bits(prepared.distances(&queries));
                for pool in &pools {
                    let fresh = pool
                        .install(|| ca_core::distance::cloud_to_cloud_par(&queries, &reference))
                        .unwrap();
                    let retained = pool.install(|| prepared.distances_par(&queries));
                    assert_eq!(bits(fresh), serial);
                    assert_eq!(bits(retained), serial);
                    assert_eq!(prepared.retained_bytes(), bytes);
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn missing_hits_and_concurrent_readers_keep_call_local_state() {
    use std::sync::Arc;
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PreparedC2cReference>();
    let references = [
        fixture(33, 0., 1.),
        vec![[f64::NAN; 3], [f64::INFINITY; 3], [0.; 3]],
        vec![[f64::NAN; 3], [f64::INFINITY; 3]],
    ];
    let queries = [
        [0.; 3],
        [-0.; 3],
        [f64::NAN, 0., 0.],
        [f64::INFINITY; 3],
        [1e308; 3],
        [0.5; 3],
    ]
    .repeat(700);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for reference in references {
        let expected = pool
            .install(|| ca_core::distance::cloud_to_cloud_par(&queries, &reference))
            .unwrap();
        let expected = bits(expected);
        let prepared = Arc::new(PreparedC2cReference::new(&reference).unwrap());
        let bytes = prepared.retained_bytes();
        std::thread::scope(|scope| {
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let prepared = Arc::clone(&prepared);
                    let queries = &queries;
                    let pool = &pool;
                    scope.spawn(move || pool.install(|| prepared.distances_par(queries)))
                })
                .collect();
            for reader in readers {
                assert_eq!(bits(reader.join().unwrap()), expected);
            }
        });
        assert_eq!(prepared.retained_bytes(), bytes);
    }
}
