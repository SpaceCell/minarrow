// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! # Concurrent stress test for the categorical dictionary
//!
//! Runs a multi-threaded intern workload against `CategoricalArray<u32>`
//! and prints throughput. Builds in three configurations:
//!
//! ```ignore
//! cargo test --release                                  --test categorical_dict_stress -- --nocapture
//! cargo test --release --features shared_dict           --test categorical_dict_stress -- --nocapture
//! cargo test --release --features fast_dict        --test categorical_dict_stress -- --nocapture
//! ```
//!
//! Without `shared_dict`, every thread holds its own `CategoricalArray`
//! with its own `Vec64<String>` dictionary; interning is a linear scan
//! of the per-thread dictionary. With `shared_dict`, every thread's
//! categorical shares one `Dictionary` so codes mean the same string
//! across all threads, and intern is a sharded hashmap lookup. With
//! `fast_dict`, the shared dictionary swaps in `parking_lot`,
//! `ahash`, `hashbrown` for higher throughput under contention.
//!
//! ## Knobs
//!
//! - `STRESS_THREADS=N` - number of worker threads (default 8).
//! - `STRESS_ITERS=N` - iterations per thread (default 500_000).
//! - `STRESS_POOL_SIZE=N` - cap on unique strings (no cap by default).
//!   When set, the dictionary cardinality is bounded by N in every
//!   feature configuration, so the comparison is apples-to-apples
//!   regardless of feature.
//!
//! ```ignore
//! STRESS_POOL_SIZE=1000 STRESS_THREADS=8 STRESS_ITERS=2000000 \
//!   cargo test --release --features fast_dict \
//!   --test categorical_dict_stress -- --nocapture
//! ```
//!
//! Note: without `shared_dict`, intern cost is O(dictionary size)
//! because each thread does a linear scan of its `Vec64<String>`.
//! Setting `STRESS_ITERS` high with no `STRESS_POOL_SIZE` grows the
//! per-thread dict to sizes where the linear scan dominates; pick a
//! `STRESS_POOL_SIZE` cap for comparable wall times.
//!
//! ## Workload model
//!
//! Each thread runs `iters_per_thread` interns. Each intern picks a
//! string via a deterministic per-thread `xorshift32` PRNG:
//!
//! - **90 %** cache hits from a 100-string seed pool.
//! - **9 %** thread-local novels (`t{thread}_n{iter}`, unique per
//!   thread per iteration).
//! - **1 %** shared novels driven by a global `AtomicUsize` counter
//!   with a window of 100. With `shared_dict` on, every 100 fetch_add
//!   calls roll to a new shared string so ~100 intern calls across
//!   threads land on the same string before moving on - the genuine
//!   same-value race window. Without `shared_dict`, threads don't
//!   share a dictionary so these reduce to per-thread novels with a
//!   coarser cycling pattern.
//!
//! ## Correctness checks (`shared_dict` configurations only)
//!
//! 1. No duplicate strings in the dictionary.
//! 2. Every published code round-trips through `lookup` to the right
//!    string.

#[cfg(feature = "shared_dict")]
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use minarrow::CategoricalArray;

#[cfg(feature = "fast_dict")]
use minarrow::Dictionary;
#[cfg(all(feature = "shared_dict", not(feature = "fast_dict")))]
use minarrow::Dictionary;

#[cfg(feature = "fast_dict")]
const MODE: &str = "fast_dict";
#[cfg(all(feature = "shared_dict", not(feature = "fast_dict")))]
const MODE: &str = "shared_dict";
#[cfg(not(feature = "shared_dict"))]
const MODE: &str = "no_shared_dict (per-thread linear-scan dict)";

/// Default per-thread iteration count when `STRESS_ITERS` is unset.
const DEFAULT_STRESS_ITERS: usize = 500_000;

/// Per-thread iteration count, controlled by `STRESS_ITERS`. Defaults
/// to `DEFAULT_STRESS_ITERS`.
fn stress_iters() -> usize {
    std::env::var("STRESS_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STRESS_ITERS)
}

/// Caps the dictionary at N unique strings via `STRESS_POOL_SIZE=N`.
/// Applies in all feature configurations. Unset = no cap.
fn pool_cap() -> Option<u32> {
    std::env::var("STRESS_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
}

/// Deterministic PRNG so each run produces the same workload mix and
/// timings are comparable across configurations.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Builds a `CategoricalArray<u32>` for the given thread.
///
/// With `shared_dict`, the array adopts the shared dictionary so every
/// thread's categorical points at the same store. Without, every thread
/// gets a fresh independent dictionary.
#[cfg(feature = "shared_dict")]
fn new_array_for_thread(
    capacity: usize,
    shared: &Dictionary<u32>,
) -> CategoricalArray<u32> {
    use vec64::Vec64;
    CategoricalArray::<u32>::new_existing_dict(
        Vec64::<u32>::with_capacity(capacity),
        shared.clone(),
        None,
    )
}

#[cfg(not(feature = "shared_dict"))]
fn new_array_for_thread(capacity: usize) -> CategoricalArray<u32> {
    CategoricalArray::<u32>::with_capacity(capacity, None, false)
}

fn run_stress(threads: usize, iters: usize) {
    const DEFAULT_SEED_POOL_SIZE: u32 = 100;
    /// How many shared-novel intern calls land on the same string
    /// before the counter rolls to the next. 100 across many threads
    /// gives a clear same-value race window (under `shared_dict`).
    const SHARED_WINDOW: usize = 100;

    // When `STRESS_POOL_SIZE=N` is set, the workload collapses to
    // 100 % cache hits over an N-string pool. Used to cap dictionary
    // cardinality at a known value for capped-dict benchmarks.
    let capped_pool = pool_cap();
    let seed_pool_size = capped_pool.unwrap_or(DEFAULT_SEED_POOL_SIZE);

    let shared_counter: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

    #[cfg(feature = "shared_dict")]
    let shared_dict: Dictionary<u32> = Dictionary::new();

    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let shared_counter = Arc::clone(&shared_counter);
            #[cfg(feature = "shared_dict")]
            let shared_dict = shared_dict.clone();

            thread::spawn(move || {
                #[cfg(feature = "shared_dict")]
                let mut cat = new_array_for_thread(iters, &shared_dict);
                #[cfg(not(feature = "shared_dict"))]
                let mut cat = new_array_for_thread(iters);

                // Distinct non-zero seed per thread via golden-ratio
                // multiplier. Avoids the all-zero degenerate state.
                let mut rng = ((t as u32).wrapping_mul(0x9E3779B9)) | 1;
                let mut cache_hits = 0u64;
                let mut thread_novels = 0u64;
                let mut shared_novels = 0u64;

                for i in 0..iters {
                    let r = xorshift(&mut rng) % 100;
                    let value = if r < 90 {
                        cache_hits += 1;
                        let pool_id = xorshift(&mut rng) % seed_pool_size;
                        format!("seed_{}", pool_id)
                    } else if r < 99 {
                        thread_novels += 1;
                        match capped_pool {
                            // Capped: draw from the same N-element
                            // pool so the dictionary is bounded by N
                            // in every feature configuration.
                            Some(cap) => {
                                let pool_id = xorshift(&mut rng) % cap;
                                format!("seed_{}", pool_id)
                            }
                            // Uncapped: thread-local unique novels;
                            // dict grows unboundedly.
                            None => format!("t{}_n{}", t, i),
                        }
                    } else {
                        shared_novels += 1;
                        match capped_pool {
                            // Capped: draw from the same N-element
                            // pool.
                            Some(cap) => {
                                let pool_id = xorshift(&mut rng) % cap;
                                format!("seed_{}", pool_id)
                            }
                            // Uncapped: shared-window novel string
                            // driven by the global counter.
                            None => {
                                let n = shared_counter.fetch_add(1, Ordering::Relaxed);
                                format!("shared_n{}", n / SHARED_WINDOW)
                            }
                        }
                    };
                    let _code = cat.push_str(&value);
                }
                (cache_hits, thread_novels, shared_novels, cat)
            })
        })
        .collect();

    let mut total_hits = 0u64;
    let mut total_thread_novels = 0u64;
    let mut total_shared_novels = 0u64;
    let mut cats: Vec<CategoricalArray<u32>> = Vec::with_capacity(threads);
    for h in handles {
        let (hits, novels, shared, cat) = h.join().expect("worker thread panicked");
        total_hits += hits;
        total_thread_novels += novels;
        total_shared_novels += shared;
        cats.push(cat);
    }
    let elapsed = start.elapsed();

    let total = total_hits + total_thread_novels + total_shared_novels;
    let throughput = total as f64 / elapsed.as_secs_f64();
    let ns_per_intern = elapsed.as_nanos() as f64 / total as f64;

    // Dictionary cardinality reporting.
    //
    // Under `shared_dict`, every thread's categorical points at the
    // same shared dictionary, so `cats[0].unique_values().len()` is the true
    // unique cardinality across all threads. Summing per-thread would
    // count each entry N times.
    //
    // Without `shared_dict`, each thread holds its own independent
    // dictionary. Sum the per-thread sizes (legitimate because they
    // are separate dicts; per-thread = total / threads).
    #[cfg(feature = "shared_dict")]
    let unique_in_dict = cats[0].unique_values().len();
    #[cfg(not(feature = "shared_dict"))]
    let unique_in_dict: usize = cats.iter().map(|c| c.unique_values().len()).sum();

    println!();
    println!("=== Categorical stress [{}] ===", MODE);
    println!("  threads                    : {}", threads);
    println!("  iters/thread               : {}", iters);
    println!("  total interns              : {}", total);
    if let Some(cap) = capped_pool {
        println!("  STRESS_POOL_SIZE cap       : {}", cap);
    }
    println!(
        "    cache hits               : {:>10} ({:>5.1}%)",
        total_hits,
        100.0 * total_hits as f64 / total as f64
    );
    println!(
        "    thread-local novels      : {:>10} ({:>5.1}%)",
        total_thread_novels,
        100.0 * total_thread_novels as f64 / total as f64
    );
    println!(
        "    shared-window novels     : {:>10} ({:>5.2}%)",
        total_shared_novels,
        100.0 * total_shared_novels as f64 / total as f64
    );
    println!("  elapsed                    : {:?}", elapsed);
    println!(
        "  throughput                 : {:>10.0} interns/sec  ({:>5.1} ns/intern)",
        throughput, ns_per_intern
    );
    #[cfg(feature = "shared_dict")]
    println!("  unique strings in dict     : {} (one shared dict across threads)", unique_in_dict);
    #[cfg(not(feature = "shared_dict"))]
    println!(
        "  unique strings in dict     : {} (sum of {} independent per-thread dicts)",
        unique_in_dict, threads
    );

    // ---- correctness verification (shared_dict configurations) ----

    #[cfg(feature = "shared_dict")]
    {
        // Every thread's categorical shares the same dictionary; pick
        // any one and inspect it.
        let dict_values = cats[0].unique_values();

        // (1) No duplicate strings.
        let unique: HashSet<&str> = dict_values.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            unique.len(),
            dict_values.len(),
            "duplicate strings in dict: same-value race leaked codes ({} values, {} unique)",
            dict_values.len(),
            unique.len(),
        );

        // (2) Every shared-window string that any thread reached should
        // exist in the dict, and resolve back to itself. Skipped under
        // a capped pool because the workload produces no shared novels.
        if capped_pool.is_none() {
            let max_shared_window = shared_counter.load(Ordering::Relaxed) / SHARED_WINDOW;
            let mut shared_seen = 0usize;
            for w in 0..=max_shared_window {
                let s = format!("shared_n{}", w);
                if let Some(code) = shared_dict.lookup(&s) {
                    let resolved = &dict_values[code as usize];
                    assert_eq!(
                        resolved, &s,
                        "code {} resolves to '{}' but should be '{}'",
                        code, resolved, s
                    );
                    shared_seen += 1;
                }
            }
            assert!(
                shared_seen > 0,
                "no shared-window strings interned; workload mix is broken"
            );
        }

        // (2b) Under a capped pool, the dictionary cardinality must
        // not exceed the cap.
        if let Some(cap) = capped_pool {
            assert!(
                dict_values.len() <= cap as usize,
                "dict has {} entries but pool cap was {}",
                dict_values.len(),
                cap
            );
        }

        // (3) Every published code resolves back to its string.
        for (i, s) in dict_values.iter().enumerate() {
            let code = shared_dict.lookup(s).unwrap_or_else(|| {
                panic!("interned value '{}' (code {}) not findable via lookup", s, i)
            });
            assert_eq!(
                code as usize, i,
                "code mismatch: lookup('{}') = {}, expected {}",
                s, code, i
            );
        }
    }

    // Suppress unused-variable warnings without `shared_dict`.
    let _ = cats;
    let _ = shared_counter;
}

/// Thread count, controlled by `STRESS_THREADS`. Defaults to 8.
fn stress_threads() -> usize {
    std::env::var("STRESS_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
}

#[test]
fn stress() {
    run_stress(stress_threads(), stress_iters());
}
