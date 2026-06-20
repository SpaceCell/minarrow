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

use minarrow::traits::selection::{ColumnSelection, RowSelection};
use minarrow::*;

fn main() {
    // Create a sample table
    let table = create_sample_table();

    println!("Original table:");
    println!("{}\n", table);

    // Table-specific API
    println!("=== Column and Row Selection ===\n");

    println!("table.c(&[\"name\", \"age\"]).r(1..4)");
    let view1 = table.c(&["name", "age"]).r(1..4);
    println!("{}\n", view1);

    println!("table.c(&[0, 2]).r(&[0, 2, 4])");
    let view2 = table.c(&[0, 2]).r(&[0, 2, 4]);
    println!("{}\n", view2);

    println!("table.c(0..2).r(0..5)");
    let view3 = table.c(0..2).r(0..5);
    println!("{}\n", view3);

    println!("table.c(1).r(&[2, 4, 6])");
    let view4 = table.c(1).r(&[2, 4, 6]);
    println!("{}\n", view4);

    // Aliased API
    println!("=== col() alias for c() ===\n");

    println!("table.col(\"id\").r(0..3)");
    let view5 = table.col("id").r(0..3);
    println!("{}\n", view5);

    println!("table.col(\"name\").r(1..5)");
    let view6 = table.col("name").r(1..5);
    println!("{}\n", view6);

    // Materialise selections
    println!("=== Materialisation ===\n");

    println!("table.c(&[\"name\", \"age\"]).r(0..3).to_table()");
    let view = table.c(&["name", "age"]).r(0..3);
    let materialised = view.to_table();
    println!("{}\n", materialised);

    // Array and FieldArray selection
    println!("=== Array & FieldArray Selection ===\n");

    // Get a single column as ArrayV via col_ix
    let age_view = table.col("age").col_ix(0).unwrap();
    println!("Age column (ArrayV):");
    println!("  Length: {}", age_view.len());
    println!(
        "  Values: {:?}\n",
        (0..age_view.len())
            .map(|i| age_view.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    // ArrayV selection
    let id_array = Array::from_int32({
        let mut arr = IntegerArray::<i32>::default();
        for i in 0..10 {
            arr.push(i * 10);
        }
        arr
    });
    let id_view = ArrayV::from(id_array);
    println!("ArrayV:");
    println!("  Length: {}", id_view.len());
    println!(
        "  Values: {:?}\n",
        (0..id_view.len())
            .map(|i| id_view.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    // Select from ArrayV using .r()
    println!("id_view.r(0..5)");
    let id_selected1 = id_view.r(0..5);
    println!("  Length: {}", id_selected1.len());
    println!(
        "  Values: {:?}\n",
        (0..id_selected1.len())
            .map(|i| id_selected1.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    println!("id_view.r(&[2, 4, 6, 8])");
    let id_selected2 = id_view.r(&[2, 4, 6, 8]);
    println!("  Length: {}", id_selected2.len());
    println!(
        "  Values: {:?}",
        (0..id_selected2.len())
            .map(|i| id_selected2.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );
}

fn create_sample_table() -> Table {
    // Create id column
    let mut id_arr = IntegerArray::<i32>::default();
    for i in 0..10 {
        id_arr.push(i);
    }

    // Create name column
    let mut name_arr = StringArray::<u32>::default();
    let names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
    ];
    for name in names {
        name_arr.push(name.to_string());
    }

    // Create age column
    let mut age_arr = IntegerArray::<i32>::default();
    for i in 0..10 {
        age_arr.push(20 + (i * 3) as i32);
    }

    let col_id = FieldArray::from_arr("id", Array::from_int32(id_arr));
    let col_name = FieldArray::from_arr("name", Array::from_string32(name_arr));
    let col_age = FieldArray::from_arr("age", Array::from_int32(age_arr));

    Table::new("People".to_string(), Some(vec![col_id, col_name, col_age]))
}
