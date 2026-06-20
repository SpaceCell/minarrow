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

//! # Routing Module
//!
//! Intelligent routing and dispatching for kernel operations
//! including broadcasting, type promotion, and operation dispatch.

pub mod arithmetic;
pub mod binary_map;
pub mod broadcast;

pub use arithmetic::resolve_binary_arithmetic;
pub use binary_map::binary_map;
pub use broadcast::{broadcast_length_1_array, maybe_broadcast_scalar_array};
