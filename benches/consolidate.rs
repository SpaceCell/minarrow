// Copyright 2025 Peter Garfield Bower
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

//! Benchmark: arena vs concat consolidation for SuperTable.
//!
//! Consolidates 100 tables of 10,000 rows each into a single table,
//! comparing the arena (single allocation) path against the per-column
//! concat fold. Two table widths are tested: 4 columns and 20 columns.
//!
//! Improvement is generally 15-20% for the arena in this single-threaded
//! case, with additional benefits from reducing allocator lock contention
//! in multi-threaded scenarios.
//!
//! Run with:
//!   cargo bench --bench consolidate --features chunked,arena

use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use minarrow::{
    Array, ArrowType, BooleanArray, Field, FieldArray, FloatArray, IntegerArray, MaskedArray,
    StringArray, SuperTable, Table,
};

const N_TABLES: usize = 100;
const ROWS_PER_TABLE: usize = 10000;

/// Build a single table with the requested number of columns.
///
/// Cycles through int64, float64, string, boolean column types so that
/// every type gets exercised regardless of width.
fn build_table(n_cols: usize, seed: usize) -> Table {
    let base = seed * ROWS_PER_TABLE;

    let cols: Vec<FieldArray> = (0..n_cols)
        .map(|col| {
            let name = format!("c{col}");
            match col % 4 {
                0 => {
                    let mut arr = IntegerArray::<i64>::with_capacity(ROWS_PER_TABLE, false);
                    for i in 0..ROWS_PER_TABLE {
                        arr.push((base + i + col) as i64);
                    }
                    FieldArray::new(
                        Field::new(&name, ArrowType::Int64, false, None),
                        Array::from(arr),
                    )
                }
                1 => {
                    let mut arr = FloatArray::<f64>::with_capacity(ROWS_PER_TABLE, false);
                    for i in 0..ROWS_PER_TABLE {
                        arr.push((base + i + col) as f64 * 0.1);
                    }
                    FieldArray::new(
                        Field::new(&name, ArrowType::Float64, false, None),
                        Array::from(arr),
                    )
                }
                2 => {
                    let mut arr = StringArray::<u32>::with_capacity(
                        ROWS_PER_TABLE,
                        ROWS_PER_TABLE * 8,
                        false,
                    );
                    for i in 0..ROWS_PER_TABLE {
                        arr.push(format!("r{}", base + i + col));
                    }
                    FieldArray::new(
                        Field::new(&name, ArrowType::String, false, None),
                        Array::from(arr),
                    )
                }
                _ => {
                    let mut arr = BooleanArray::with_capacity(ROWS_PER_TABLE, false);
                    for i in 0..ROWS_PER_TABLE {
                        arr.push((base + i + col) % 3 == 0);
                    }
                    FieldArray::new(
                        Field::new(&name, ArrowType::Boolean, false, None),
                        Array::from(arr),
                    )
                }
            }
        })
        .collect();

    Table {
        cols,
        n_rows: ROWS_PER_TABLE,
        name: "batch".into(),
        ..Default::default()
    }
}

fn build_super_table(n_cols: usize) -> SuperTable {
    let batches: Vec<Arc<Table>> = (0..N_TABLES)
        .map(|i| Arc::new(build_table(n_cols, i)))
        .collect();
    SuperTable::from_batches(batches, Some("bench".into()))
}

/// Concat-fold consolidation, inlined from `SuperTable::consolidate_concat`.
///
/// The actual implementation is private so it cannot be called from an
/// external benchmark crate. We reproduce the same logic here to get an
/// apples-to-apples comparison against the arena path exposed via
/// `Consolidate::consolidate`.
fn consolidate_concat(st: SuperTable) -> Table {
    let n_cols = st.schema.len();
    let mut cols = Vec::with_capacity(n_cols);

    for col_idx in 0..n_cols {
        let field = st.schema[col_idx].clone();
        let mut arr = st.batches[0].cols[col_idx].array.clone();
        for batch in st.batches.iter().skip(1) {
            arr.concat_array(&batch.cols[col_idx].array);
        }
        let null_count = arr.null_count();
        cols.push(FieldArray {
            field,
            array: arr,
            null_count,
        });
    }

    Table {
        cols,
        n_rows: st.n_rows,
        name: st.name,
        ..Default::default()
    }
}

fn bench_consolidate(c: &mut Criterion) {
    use minarrow::Consolidate;

    for n_cols in [4, 20] {
        let mut group = c.benchmark_group(format!("consolidate_{n_cols}_cols"));

        group.bench_function("concat", |b| {
            b.iter_with_setup(
                || build_super_table(n_cols),
                |st| black_box(consolidate_concat(st)),
            )
        });

        group.bench_function("arena", |b| {
            b.iter_with_setup(
                || build_super_table(n_cols),
                |st| black_box(st.consolidate()),
            )
        });

        group.finish();
    }
}

criterion_group!(benches, bench_consolidate);
criterion_main!(benches);
