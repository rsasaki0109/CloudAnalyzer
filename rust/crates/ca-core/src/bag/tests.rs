use super::*;
use std::io::Cursor;

/// Serialises ROS 2 messages (CDR, little-endian) the way a bag holds them.
struct Cdr(Vec<u8>);

impl Cdr {
    fn new() -> Self {
        Cdr(vec![0, 1, 0, 0])
    }

    fn align(&mut self, n: usize) {
        while !(self.0.len() - 4).is_multiple_of(n) {
            self.0.push(0);
        }
    }

    fn u8(mut self, v: u8) -> Self {
        self.0.push(v);
        self
    }

    fn u32(mut self, v: u32) -> Self {
        self.align(4);
        self.0.extend(v.to_le_bytes());
        self
    }

    fn f64s(mut self, values: &[f64]) -> Self {
        for v in values {
            self.align(8);
            self.0.extend(v.to_le_bytes());
        }
        self
    }

    fn string(mut self, s: &str) -> Self {
        self = self.u32(s.len() as u32 + 1);
        self.0.extend(s.bytes());
        self.0.push(0);
        self
    }

    fn raw(mut self, b: &[u8]) -> Self {
        self = self.u32(b.len() as u32);
        self.0.extend(b);
        self
    }

    fn header(self, time: f64) -> Self {
        self.u32(time as u32)
            .u32(((time % 1.0) * 1e9).round() as u32)
            .string("lidar")
    }
}

fn point_cloud(points: &[[f32; 4]], time: f64) -> Vec<u8> {
    let mut data = Vec::new();
    for p in points {
        for v in p {
            data.extend(v.to_le_bytes());
        }
    }
    let mut m = Cdr::new()
        .header(time)
        .u32(1)
        .u32(points.len() as u32)
        .u32(4);
    for (k, name) in ["x", "y", "z", "intensity"].iter().enumerate() {
        m = m.string(name).u32(4 * k as u32).u8(7).u32(1);
    }
    m.u8(0)
        .u32(16)
        .u32(16 * points.len() as u32)
        .raw(&data)
        .u8(1)
        .0
}

fn imu(yaw: f64, time: f64) -> Vec<u8> {
    Cdr::new()
        .header(time)
        .f64s(&[0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()])
        .f64s(&[0.0; 9 + 3 + 9])
        .f64s(&[0.0, 0.0, 9.8])
        .f64s(&[0.0; 9])
        .0
}

#[test]
fn raw_imu_retains_exact_time_and_all_fields() {
    let data = Cdr::new()
        .u32(1_700_000_000)
        .u32(123)
        .string("imu/data")
        .f64s(&[0.0, 0.0, 0.0, 1.0])
        .f64s(&[0.0; 9])
        .f64s(&[0.1, -0.2, 0.3])
        .f64s(&[0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03])
        .f64s(&[1.0, 2.0, 9.81])
        .f64s(&[-1.0; 9])
        .0;
    let raw = decode_imu_raw(&data, Encoding::Cdr).unwrap();
    assert_eq!(raw.stamp_ns, 1_700_000_000_000_000_123);
    assert_eq!(raw.frame_id, "imu/data");
    assert_eq!(raw.angular_velocity, [0.1, -0.2, 0.3]);
    assert_eq!(raw.angular_velocity_covariance[8], 0.03);
    assert_eq!(raw.acceleration, [1.0, 2.0, 9.81]);
    assert_eq!(raw.orientation_status(), MeasurementStatus::Unknown);
    assert_eq!(raw.angular_velocity_status(), MeasurementStatus::Known);
    assert_eq!(raw.acceleration_status(), MeasurementStatus::Missing);
    assert!(decode_imu_raw(&data[..data.len() - 1], Encoding::Cdr).is_err());
    assert_eq!(
        decode_imu(&data, Encoding::Cdr).unwrap().up,
        Some([0.0, 0.0, 1.0])
    );
}

#[test]
fn raw_imu_handles_signed_ros2_seconds_and_rejects_bad_nanoseconds() {
    let mut data = imu(0.0, 0.0);
    data[4..8].copy_from_slice(&(-1_i32).to_le_bytes());
    data[8..12].copy_from_slice(&999_999_999_u32.to_le_bytes());
    let raw = decode_imu_raw(&data, Encoding::Cdr).unwrap();
    assert_eq!(raw.stamp_ns, -1);
    data[8..12].copy_from_slice(&1_000_000_000_u32.to_le_bytes());
    assert!(decode_imu_raw(&data, Encoding::Cdr).is_err());
}

#[test]
fn imu_covariance_status_does_not_confuse_missing_unknown_or_invalid() {
    use MeasurementStatus::*;
    assert_eq!(measurement_status(&[1.0; 3], &[0.0; 9]), Unknown);
    assert_eq!(measurement_status(&[f64::NAN; 3], &[-1.0; 9]), Missing);
    assert_eq!(measurement_status(&[f64::NAN; 3], &[0.0; 9]), Invalid);
    for covariance in [
        [-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, -0.9, -0.9, -0.9, 1.0, -0.9, -0.9, -0.9, 1.0],
    ] {
        assert_eq!(measurement_status(&[1.0; 3], &covariance), Invalid);
    }
    assert_eq!(measurement_status(&[1.0; 3], &[1.0; 9]), Known);
    let mut raw = decode_imu_raw(&imu(0.0, 1.0), Encoding::Cdr).unwrap();
    raw.orientation = [0.0; 4];
    assert_eq!(raw.orientation_status(), Invalid);
    raw.orientation_covariance[0] = -1.0;
    assert_eq!(raw.orientation_status(), Missing);
}

/// An MCAP (unchunked, without a summary) of `messages` on channels 0 (points) and 1 (imu).
fn mcap(messages: &[(u16, f64, Vec<u8>)]) -> Vec<u8> {
    let record = |op: u8, body: &[u8]| {
        let mut r = vec![op];
        r.extend((body.len() as u64).to_le_bytes());
        r.extend(body);
        r
    };
    let string = |s: &str| {
        let mut b = (s.len() as u32).to_le_bytes().to_vec();
        b.extend(s.bytes());
        b
    };
    let magic = b"\x89MCAP0\r\n";
    let mut out = magic.to_vec();
    out.extend(record(0x01, &[string("ros2"), string("test")].concat()));
    for (id, (topic, kind)) in [
        ("/points", "sensor_msgs/msg/PointCloud2"),
        ("/imu", "sensor_msgs/msg/Imu"),
    ]
    .iter()
    .enumerate()
    {
        let id = id as u16;
        out.extend(record(
            0x03,
            &[
                (id + 1).to_le_bytes().to_vec(),
                string(kind),
                string("ros2msg"),
                0u32.to_le_bytes().to_vec(),
            ]
            .concat(),
        ));
        out.extend(record(
            0x04,
            &[
                id.to_le_bytes().to_vec(),
                (id + 1).to_le_bytes().to_vec(),
                string(topic),
                string("cdr"),
                0u32.to_le_bytes().to_vec(),
            ]
            .concat(),
        ));
    }
    for (k, (channel, time, data)) in messages.iter().enumerate() {
        let ns = (time * 1e9).round() as u64;
        out.extend(record(
            0x05,
            &[
                channel.to_le_bytes().to_vec(),
                (k as u32).to_le_bytes().to_vec(),
                ns.to_le_bytes().to_vec(),
                ns.to_le_bytes().to_vec(),
                data.clone(),
            ]
            .concat(),
        ));
    }
    out.extend(record(0x0f, &0u32.to_le_bytes()));
    out.extend(record(0x02, &[0u8; 20]));
    out.extend(magic);
    out
}

/// A ROS 1 bag (format 2.0) with one chunk of `messages` on connection 0 (points) and 1 (imu).
fn ros1_bag(messages: &[(u32, f64, Vec<u8>)], compression: &str) -> Vec<u8> {
    let field = |name: &str, value: &[u8]| {
        let mut f = ((name.len() + 1 + value.len()) as u32)
            .to_le_bytes()
            .to_vec();
        f.extend(name.bytes());
        f.push(b'=');
        f.extend(value);
        f
    };
    let time = |t: f64| {
        [
            (t as u32).to_le_bytes(),
            (((t % 1.0) * 1e9).round() as u32).to_le_bytes(),
        ]
        .concat()
    };
    let record = |header: &[Vec<u8>], data: &[u8]| {
        let header = header.concat();
        let mut r = (header.len() as u32).to_le_bytes().to_vec();
        r.extend(&header);
        r.extend((data.len() as u32).to_le_bytes());
        r.extend(data);
        r
    };
    // The ROS 1 messages: CDR data works for the test's decoders too, so the
    // chunk holds CDR bytes tagged as ROS 1; the reader does not look inside.
    let mut chunk_records = Vec::new();
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for (conn, t, data) in messages {
        chunk_records.extend(record(
            &[
                field("op", &[2]),
                field("conn", &conn.to_le_bytes()),
                field("time", &time(*t)),
            ],
            data,
        ));
        *counts.entry(*conn).or_default() += 1;
    }
    let compressed = match compression {
        "lz4" => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
            std::io::Write::write_all(&mut enc, &chunk_records).unwrap();
            enc.finish().unwrap()
        }
        "bz2" => {
            let mut enc = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
            std::io::Write::write_all(&mut enc, &chunk_records).unwrap();
            enc.finish().unwrap()
        }
        _ => chunk_records.clone(),
    };
    let mut out = b"#ROSBAG V2.0\n".to_vec();
    let header_at = out.len();
    // The bag header is padded to a fixed size so that index_pos can be filled in.
    let header_record = |index_pos: u64| {
        let fields = [
            field("op", &[3]),
            field("index_pos", &index_pos.to_le_bytes()),
            field("conn_count", &2u32.to_le_bytes()),
            field("chunk_count", &1u32.to_le_bytes()),
        ]
        .concat();
        let mut r = (fields.len() as u32).to_le_bytes().to_vec();
        r.extend(&fields);
        r.extend(0u32.to_le_bytes());
        r
    };
    out.extend(header_record(0));
    let chunk_at = out.len() as u64;
    out.extend(record(
        &[
            field("op", &[5]),
            field("compression", compression.as_bytes()),
            field("size", &(chunk_records.len() as u32).to_le_bytes()),
        ],
        &compressed,
    ));
    let index_at = out.len() as u64;
    for (conn, (topic, kind)) in [
        ("/points", "sensor_msgs/PointCloud2"),
        ("/imu", "sensor_msgs/Imu"),
    ]
    .iter()
    .enumerate()
    {
        let info = [
            field("topic", topic.as_bytes()),
            field("type", kind.as_bytes()),
            field("md5sum", b"0"),
            field("message_definition", b""),
        ]
        .concat();
        out.extend(record(
            &[
                field("op", &[7]),
                field("conn", &(conn as u32).to_le_bytes()),
                field("topic", topic.as_bytes()),
            ],
            &info,
        ));
    }
    let mut count_data = Vec::new();
    for (conn, n) in &counts {
        count_data.extend(conn.to_le_bytes());
        count_data.extend(n.to_le_bytes());
    }
    out.extend(record(
        &[
            field("op", &[6]),
            field("ver", &1u32.to_le_bytes()),
            field("chunk_pos", &chunk_at.to_le_bytes()),
            field("start_time", &time(messages[0].1)),
            field("end_time", &time(messages.last().unwrap().1)),
            field("count", &(counts.len() as u32).to_le_bytes()),
        ],
        &count_data,
    ));
    let header = header_record(index_at);
    out[header_at..header_at + header.len()].copy_from_slice(&header);
    out
}

fn scans() -> Vec<(u16, f64, Vec<u8>)> {
    (0..3)
        .flat_map(|k| {
            let t = 100.0 + 0.1 * k as f64;
            let points: Vec<[f32; 4]> = (0..5).map(|i| [i as f32, k as f32, 1.0, 0.5]).collect();
            [
                (1, t - 0.01, imu(0.1 * k as f64, t - 0.01)),
                (0, t, point_cloud(&points, t)),
            ]
        })
        .collect()
}

fn check<R: Read + Seek>(mut bag: Bag<R>) {
    let names: Vec<(String, String, u64)> = bag
        .topics()
        .iter()
        .map(|t| (t.name.clone(), t.kind.clone(), t.count))
        .collect();
    assert_eq!(
        names,
        [
            ("/imu".into(), IMU.into(), 3),
            ("/points".into(), POINT_CLOUD.into(), 3)
        ]
    );
    let messages: Vec<Message> = bag
        .messages(&["/points", "/imu"])
        .map(|m| m.unwrap())
        .collect();
    assert_eq!(messages.len(), 6);
    let mut clouds = Vec::new();
    let mut imus = Vec::new();
    for m in &messages {
        match m.topic.as_str() {
            "/points" => clouds.push(decode_point_cloud2(&m.data, m.encoding).unwrap()),
            _ => imus.push(decode_imu(&m.data, m.encoding).unwrap()),
        }
    }
    assert_eq!(clouds.len(), 3);
    assert_eq!(clouds[2].positions[4], [4.0, 2.0, 1.0]);
    assert_eq!(clouds[2].intensity.as_deref(), Some(&[0.5f32; 5][..]));
    assert!(clouds[2].time.is_none());
    assert!((clouds[1].stamp - 100.1).abs() < 1e-6);
    let ups = ups_at(&imus, &[100.2], 0.5);
    let up = ups[0].unwrap();
    assert!((up[2] - 1.0).abs() < 1e-9 && up[0].abs() < 1e-9);
    // Only one topic asked for.
    assert_eq!(bag.messages(&["/imu"]).count(), 3);
}

#[test]
fn an_mcap_is_read() {
    check(Bag::from_reader(Cursor::new(mcap(&scans()))).unwrap());
}

#[test]
fn a_ros1_bag_is_read_compressed_or_not() {
    let messages: Vec<(u32, f64, Vec<u8>)> = scans()
        .into_iter()
        .map(|(c, t, d)| (u32::from(c), t, d))
        .collect();
    for compression in ["none", "lz4", "bz2"] {
        let bytes = ros1_bag(&messages, compression);
        let mut bag = Bag::from_reader(Cursor::new(bytes)).unwrap();
        assert!(bag.is_ros1());
        // The messages are CDR bytes tagged as ROS 1 by the bag: decode them as CDR.
        let found: Vec<PointsMessage> = bag
            .messages(&["/points"])
            .map(|m| decode_point_cloud2(&m.unwrap().data, Encoding::Cdr).unwrap())
            .collect();
        assert_eq!(found.len(), 3, "{compression}");
        assert_eq!(found[1].positions[0], [0.0, 1.0, 1.0]);
    }
}

#[test]
fn equal_recording_times_preserve_physical_order_and_topic_filtering() {
    // Physical order differs from timestamp order. The four records at
    // time 1 must keep their order, including across requested topics.
    let messages = vec![
        (1, 2.0, vec![4]),
        (0, 1.0, vec![90]),
        (1, 1.0, vec![0]),
        (0, 1.0, vec![91]),
        (1, 1.0, vec![1]),
        (1, 3.0, vec![5]),
    ];
    for compression in ["none", "lz4", "bz2"] {
        let mut bag = Bag::from_reader(Cursor::new(ros1_bag(&messages, compression))).unwrap();
        for (topics, expected) in [
            (vec!["/points", "/imu"], vec![90, 0, 91, 1, 4, 5]),
            (vec!["/imu"], vec![0, 1, 4, 5]),
            (vec!["/points"], vec![90, 91]),
        ] {
            let found: Vec<_> = bag.messages(&topics).map(|m| m.unwrap()).collect();
            assert!(found.windows(2).all(|w| w[0].time <= w[1].time));
            assert_eq!(
                found.iter().map(|m| m.data[0]).collect::<Vec<_>>(),
                expected,
                "{compression}, {topics:?}"
            );
        }
    }
}

#[test]
fn a_ros1_point_cloud_decodes_with_a_time_field() {
    // ROS 1 layout: seq, stamp, frame_id (no NUL), no alignment.
    let mut b = Vec::new();
    b.extend(7u32.to_le_bytes());
    b.extend(12u32.to_le_bytes());
    b.extend(500_000_000u32.to_le_bytes());
    b.extend(3u32.to_le_bytes());
    b.extend(b"os1");
    b.extend(1u32.to_le_bytes());
    b.extend(3u32.to_le_bytes());
    b.extend(4u32.to_le_bytes());
    for (k, name) in ["x", "y", "z", "t"].iter().enumerate() {
        b.extend((name.len() as u32).to_le_bytes());
        b.extend(name.bytes());
        b.extend((4 * k as u32).to_le_bytes());
        b.push(if *name == "t" { 6 } else { 7 });
        b.extend(1u32.to_le_bytes());
    }
    b.push(0);
    b.extend(16u32.to_le_bytes());
    b.extend(48u32.to_le_bytes());
    b.extend(48u32.to_le_bytes());
    for (p, t) in [
        ([1.0f32, 2.0, 3.0], 0u32),
        ([f32::NAN, 0.0, 0.0], 50),
        ([4.0, 5.0, 6.0], 100),
    ] {
        for v in p {
            b.extend(v.to_le_bytes());
        }
        b.extend(t.to_le_bytes());
    }
    b.push(0);
    let cloud = decode_point_cloud2(&b, Encoding::Ros1).unwrap();
    assert_eq!(cloud.stamp_ns, 12_500_000_000);
    assert_eq!(cloud.frame_id, "os1");
    assert!((cloud.stamp - 12.5).abs() < 1e-9);
    assert_eq!(cloud.positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(cloud.time.as_deref(), Some(&[0.0f32, 1.0][..]));
    assert!(cloud.intensity.is_none());
    assert!(cloud.raw_time.is_none());
    let retained = decode_point_cloud2_with_raw_time(&b, Encoding::Ros1, "t").unwrap();
    assert_eq!(retained.positions, cloud.positions);
    assert_eq!(retained.time, cloud.time);
    let raw = retained.raw_time.unwrap();
    assert_eq!(raw.field, "t");
    assert_eq!(raw.datatype, 6);
    assert_eq!(raw.values, [0.0, 100.0]);
}

#[test]
fn explicit_point_time_retains_custom_nonfinite_constant_and_empty_values() {
    let values = [0.5f32, -0.0, f32::NAN, f32::INFINITY];
    let points: Vec<_> = values.iter().map(|&v| [1.0, 2.0, 3.0, v]).collect();
    let data = point_cloud(&points, 0.0);
    let cloud = decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "intensity").unwrap();
    assert!(cloud.time.is_none());
    let raw = cloud.raw_time.unwrap();
    assert_eq!(raw.field, "intensity");
    assert_eq!(raw.datatype, 7);
    assert_eq!(
        raw.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        values.map(|v| f64::from(v).to_bits())
    );
    for points in [
        vec![[1.0, 2.0, 3.0, 7.5], [4.0, 5.0, 6.0, 7.5]],
        vec![[f32::NAN, 2.0, 3.0, 7.5]],
    ] {
        let data = point_cloud(&points, 0.0);
        let cloud = decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "intensity").unwrap();
        assert!(cloud.time.is_none());
        assert_eq!(
            cloud.raw_time.unwrap().values,
            vec![7.5; cloud.positions.len()]
        );
    }
}

#[test]
fn explicit_point_time_rejects_ambiguous_or_incomplete_layouts() {
    let make = |datatype, count, offset, row_step, bytes, duplicate| {
        let mut m = Cdr::new()
            .header(0.0)
            .u32(1)
            .u32(1)
            .u32(4 + u32::from(duplicate));
        for (k, name) in ["x", "y", "z"].iter().enumerate() {
            m = m.string(name).u32(k as u32 * 4).u8(7).u32(1);
        }
        m = m.string("t").u32(offset).u8(datatype).u32(count);
        if duplicate {
            m = m.string("t").u32(offset).u8(datatype).u32(count);
        }
        m.u8(0).u32(16).u32(row_step).raw(&vec![0; bytes]).u8(1).0
    };
    for (datatype, count, offset, row_step, bytes, duplicate) in [
        (0, 1, 12, 16, 16, false),
        (9, 1, 12, 16, 16, false),
        (6, 0, 12, 16, 16, false),
        (6, 2, 12, 16, 16, false),
        (6, 1, 13, 16, 16, false),
        (6, 1, u32::MAX, 16, 16, false),
        (6, 1, 12, 15, 16, false),
        (6, 1, 12, 16, 15, false),
        (6, 1, 12, 16, 16, true),
    ] {
        let data = make(datatype, count, offset, row_step, bytes, duplicate);
        assert!(decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "t").is_err());
    }
    let data = make(6, 1, 12, 16, 16, false);
    assert!(decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "absent").is_err());
    assert!(field_value(&[0; 4], usize::MAX, 6, false).is_none());
}

#[test]
fn point_cloud_header_retains_integer_nanoseconds_and_signed_cdr_seconds() {
    for (sec, nsec) in [
        (i32::MIN, 0u32),
        (-1, 999_999_999),
        (0, 0),
        (1_700_000_000, 123),
        (i32::MAX, 999_999_999),
    ] {
        let mut data = point_cloud(&[[1.0, 2.0, 3.0, 0.5]], 0.0);
        data[4..8].copy_from_slice(&sec.to_le_bytes());
        data[8..12].copy_from_slice(&nsec.to_le_bytes());
        let cloud = decode_point_cloud2(&data, Encoding::Cdr).unwrap();
        assert_eq!(
            cloud.stamp_ns,
            i64::from(sec) * 1_000_000_000 + i64::from(nsec)
        );
        assert_eq!(cloud.stamp, f64::from(sec) + f64::from(nsec) * 1e-9);
        assert_eq!(cloud.frame_id, "lidar");
        assert_eq!(cloud.positions, [[1.0, 2.0, 3.0]]);
    }
    let mut data = point_cloud(&[], 0.0);
    data[8..12].copy_from_slice(&1_000_000_000u32.to_le_bytes());
    assert!(
        decode_point_cloud2(&data, Encoding::Cdr)
            .unwrap_err()
            .0
            .contains("nanoseconds")
    );
}
