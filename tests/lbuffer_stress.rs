// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Concurrent correctness stress for LBuffer-backed columns
//!
//! A single writer fills a fixed-capacity window, seals it, and opens a
//! fresh one continuously for a wall-clock duration. Reader threads read
//! the current window throughout. There is no shared mutable structure and
//! no lock: column growth is published through the atomic length, and the
//! only cross-thread handoff is the replaced window, broadcast to each
//! reader over its own channel. Readers access the current window
//! lock-free and pick up the next when it arrives.
//!
//! This validates correctness, not throughput - readers deliberately
//! re-read `n_rows()` every iteration to maximise contention and expose
//! ordering bugs.
//!
//! The writer stores a known function of the global row index, so any
//! reader can verify every value it observes; a torn read, a stale base,
//! or an uninitialised slot would mismatch.
//!
//! Run:
//!
//! ```ignore
//! cargo test --release --features lbuffer --test lbuffer_stress -- --nocapture
//! ```
//!
//! ## Configuration
//!
//! - `STRESS_SECS=N`    - wall-clock seconds the writer runs (default 2).
//! - `STRESS_WINDOW=N`  - rows per window (default 1_000_000).
//! - `STRESS_READERS=N` - concurrent reader threads (default 4).
//!
//! ## Under ThreadSanitizer
//!
//! Run these tests under TSan to verify the atomic orderings on the real
//! implementation - data races and missing `Acquire`/`Release` surface as race
//! reports. Use a small window so the heap allocation path is exercised:
//!
//! ```ignore
//! RUSTFLAGS="-Zsanitizer=thread" STRESS_SECS=1 STRESS_WINDOW=131072 \
//!   cargo test -Zbuild-std --release --features lbuffer --test lbuffer_stress \
//!   --target x86_64-unknown-linux-gnu -- --nocapture --test-threads=1
//! ```
//!
//! `-Zbuild-std` instruments `std` so TSan sees its synchronisation. Weakening
//! any `Release`/`Acquire` in the mask path makes this run report a race.

#![cfg(feature = "lbuffer")]

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use minarrow::{Array, FieldArray, FloatArray, IntegerArray, LBuffer, Table};

/// Writer's value function for the price column, keyed by global row index.
#[inline]
fn price_at(row: usize) -> f64 {
    (row as f64) * 0.5 + 100.0
}

/// Writer's value function for the volume column, keyed by global row index.
#[inline]
fn volume_at(row: usize) -> i64 {
    (row as i64) * 7 + 3
}

/// Reads an environment variable as the requested type, falling back to `default`.
fn env_or<T: FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Builds a two-column `Table` backed by `LBuffer`s and returns it with the
/// writer handles for its columns.
fn open_window(window: usize) -> (LBuffer<f64>, LBuffer<i64>, Arc<Table>) {
    let price = LBuffer::<f64>::with_capacity(window);
    let volume = LBuffer::<i64>::with_capacity(window);
    let table = Arc::new(Table::new(
        "chunk".to_string(),
        Some(vec![
            FieldArray::from_arr(
                "price",
                Array::from_float64(FloatArray::<f64> {
                    data: price.as_buffer(),
                    null_mask: None,
                }),
            ),
            FieldArray::from_arr(
                "volume",
                Array::from_int64(IntegerArray::<i64> {
                    data: volume.as_buffer(),
                    null_mask: None,
                }),
            ),
        ]),
    ));
    (price, volume, table)
}

/// Verifies a window's columns against the writer's value functions over the
/// global range `[start, start + n)`.
fn verify_window(table: &Table, start: usize, n: usize) {
    let price = table.cols[0].array.num().f64();
    let volume = table.cols[1].array.num().i64();
    let price_rows = &price.data.as_slice()[..n];
    let volume_rows = &volume.data.as_slice()[..n];
    for i in 0..n {
        assert_eq!(price_rows[i], price_at(start + i), "price mismatch at global row {}", start + i);
        assert_eq!(volume_rows[i], volume_at(start + i), "volume mismatch at global row {}", start + i);
    }
}

#[test]
fn lbuffer_table_min_floor_is_deterministic() {
    // Single-threaded proof that Table::n_rows() tracks the trailing column.
    let (mut price, mut volume, table) = open_window(8);
    assert_eq!(table.n_rows(), 0);

    price.push(price_at(0)).unwrap();
    // Price is one row ahead; the floor is still the volume column at zero.
    assert_eq!(table.n_rows(), 0);

    volume.push(volume_at(0)).unwrap();
    assert_eq!(table.n_rows(), 1);

    price.push(price_at(1)).unwrap();
    assert_eq!(table.n_rows(), 1);
    volume.push(volume_at(1)).unwrap();
    assert_eq!(table.n_rows(), 2);

    verify_window(&table, 0, table.n_rows());
}

#[test]
fn lbuffer_concurrent_lock_free_reads() {
    let secs = env_or("STRESS_SECS", 2u64);
    let window = env_or("STRESS_WINDOW", 1_000_000usize);
    let readers = env_or("STRESS_READERS", 4usize);

    let done = Arc::new(AtomicBool::new(false));

    // One channel per reader carries each new window, with the global row
    // index its first row maps to.
    let mut senders = Vec::with_capacity(readers);
    let mut reader_handles = Vec::with_capacity(readers);
    for reader_id in 0..readers {
        let (tx, rx) = mpsc::channel::<(usize, Arc<Table>)>();
        senders.push(tx);
        let done = Arc::clone(&done);
        reader_handles.push(thread::spawn(move || {
            let mut current: Option<(usize, Arc<Table>)> = None;
            let mut max_global = 0usize;
            loop {
                while let Ok(window) = rx.try_recv() {
                    current = Some(window);
                }
                if let Some((start, table)) = current.as_ref() {
                    let n = table.n_rows();
                    if n > 0 {
                        let price = table.cols[0].array.num().f64();
                        let volume = table.cols[1].array.num().i64();
                        let price_slice = price.data.as_slice();
                        let volume_slice = volume.data.as_slice();
                        assert!(price_slice.len() >= n, "reader {reader_id}: price below the row floor");
                        assert!(volume_slice.len() >= n, "reader {reader_id}: volume below the row floor");
                        // Check the newest row - the slot most exposed to a
                        // torn read - plus the head row.
                        let last = start + n - 1;
                        assert_eq!(price_slice[n - 1], price_at(last), "reader {reader_id}: torn price");
                        assert_eq!(volume_slice[n - 1], volume_at(last), "reader {reader_id}: torn volume");
                        assert_eq!(price_slice[0], price_at(*start), "reader {reader_id}: head price");
                        let observed = start + n;
                        assert!(observed >= max_global, "reader {reader_id}: progress went backwards");
                        max_global = observed;
                    }
                }
                if done.load(Ordering::Acquire) {
                    break;
                }
            }
            max_global
        }));
    }

    // Writer fills and replaces windows continuously for the configured
    // duration.
    let (mut price, mut volume, mut table) = open_window(window);
    for tx in &senders {
        tx.send((0, Arc::clone(&table))).unwrap();
    }
    let mut chunk_start = 0usize;
    let mut global = 0usize;
    let start = Instant::now();
    let duration = Duration::from_secs(secs);

    'fill: loop {
        if start.elapsed() >= duration {
            break;
        }
        let mut filled = 0usize;
        while filled < window {
            // Price leads, volume trails: the table floor follows volume.
            price.push(price_at(global)).unwrap();
            volume.push(volume_at(global)).unwrap();
            global += 1;
            filled += 1;
            if filled & 0xFFFF == 0 && start.elapsed() >= duration {
                break;
            }
        }
        if filled < window {
            // Timed out mid-window: keep it as the final window.
            break 'fill;
        }
        price.seal();
        volume.seal();
        let (next_price, next_volume, next_table) = open_window(window);
        chunk_start = global;
        for tx in &senders {
            tx.send((chunk_start, Arc::clone(&next_table))).unwrap();
        }
        price = next_price;
        volume = next_volume;
        table = next_table;
    }

    price.seal();
    volume.seal();
    done.store(true, Ordering::Release);
    drop(senders);
    let total = global;

    for handle in reader_handles {
        let max_global = handle.join().unwrap();
        assert!(max_global <= total, "reader observed more rows than produced");
    }

    // Reconcile the window.
    let n = table.n_rows();
    verify_window(&table, chunk_start, n);
    assert_eq!(chunk_start + n, total, "final window does not reconcile with produced rows");
    assert!(total > 0, "stress produced no rows");

    println!("verified {total} rows with {readers} readers, no torn reads");
}

/// Whether the row at global index `i` is null, so any reader can verify it.
#[inline]
fn is_null_row(i: usize) -> bool {
    i % 7 == 0 || i % 13 == 0
}

#[test]
fn masked_concurrent_validity_reads() {
    let secs = env_or("STRESS_SECS", 2u64);
    let cap = env_or("STRESS_WINDOW", 8_000_000usize);
    let readers = env_or("STRESS_READERS", 4usize);

    // One masked window. Readers share the value buffer and the validity
    // mask, both LBuffer-backed views over the same writer.
    let mut buf = LBuffer::<i64>::with_capacity_masked(cap);
    let mask = buf.as_bitmask();
    let data = buf.as_buffer();
    let done = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::with_capacity(readers);
    for reader_id in 0..readers {
        let mask = mask.clone();
        let data = data.clone();
        let done = Arc::clone(&done);
        handles.push(thread::spawn(move || {
            let mut max_seen = 0usize;
            loop {
                // The value length is the authoritative row count: the mask
                // cursor advances before the value length is published, so
                // the mask always covers `[0, n)`.
                let n = data.len();
                assert!(n >= max_seen, "reader {reader_id}: rows went backwards");
                if n > 0 {
                    let last = n - 1;
                    assert_eq!(
                        mask.get(last),
                        !is_null_row(last),
                        "reader {reader_id}: torn validity at {last}"
                    );
                    assert_eq!(
                        mask.get(0),
                        !is_null_row(0),
                        "reader {reader_id}: head validity"
                    );
                    if !is_null_row(last) {
                        assert_eq!(
                            data.as_slice()[last],
                            last as i64,
                            "reader {reader_id}: torn value at {last}"
                        );
                    }
                }
                max_seen = n;
                if done.load(Ordering::Acquire) {
                    break;
                }
            }
            max_seen
        }));
    }

    let start = Instant::now();
    let duration = Duration::from_secs(secs);
    let mut produced = 0usize;
    while produced < cap && start.elapsed() < duration {
        if is_null_row(produced) {
            buf.push_null().unwrap();
        } else {
            buf.push(produced as i64).unwrap();
        }
        produced += 1;
    }
    buf.seal();
    done.store(true, Ordering::Release);
    for handle in handles {
        let observed = handle.join().unwrap();
        assert!(observed <= produced, "reader observed more rows than produced");
    }

    // Reconcile the window.
    assert_eq!(mask.len(), produced);
    assert_eq!(data.len(), produced);
    let mut nulls = 0usize;
    let slice = data.as_slice();
    for i in 0..produced {
        let want_null = is_null_row(i);
        assert_eq!(mask.get(i), !want_null, "final validity at {i}");
        if want_null {
            nulls += 1;
        } else {
            assert_eq!(slice[i], i as i64, "final value at {i}");
        }
    }
    assert_eq!(mask.count_zeros(), nulls, "null count mismatch");
    assert!(produced > 0, "stress produced no rows");

    println!("verified {produced} rows ({nulls} nulls) with {readers} readers, no torn validity");
}
