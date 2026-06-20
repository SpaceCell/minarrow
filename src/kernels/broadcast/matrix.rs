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

//! Matrix addition broadcasting operations (stub)
//!
//! TODO: Implement matrix broadcasting
use crate::enums::error::MinarrowError;
use std::sync::Arc;

#[cfg(feature = "matrix")]
pub fn broadcast_matrix_add(
    l: Arc<crate::Matrix>,
    r: Arc<crate::Matrix>,
) -> Result<crate::Matrix, MinarrowError> {
    let _ = (l, r);
    unimplemented!("Matrix broadcasting not yet implemented")
}

#[cfg(all(feature = "matrix", feature = "scalar_type"))]
pub fn broadcast_matrix_scalar_add(
    l: Arc<crate::Matrix>,
    r: crate::Scalar,
) -> Result<crate::Matrix, MinarrowError> {
    let _ = (l, r);
    unimplemented!("Matrix-scalar broadcasting not yet implemented")
}

#[cfg(all(feature = "matrix", feature = "scalar_type"))]
pub fn broadcast_scalar_matrix_add(
    l: crate::Scalar,
    r: Arc<crate::Matrix>,
) -> Result<crate::Matrix, MinarrowError> {
    let _ = (l, r);
    unimplemented!("Scalar-matrix broadcasting not yet implemented")
}

#[cfg(all(feature = "matrix", feature = "value_type"))]
pub fn broadcast_matrix_array_add(
    l: Arc<crate::Matrix>,
    r: Arc<crate::Array>,
) -> Result<crate::enums::value::Value, MinarrowError> {
    let _ = (l, r);
    unimplemented!("Matrix-array broadcasting not yet implemented")
}

#[cfg(all(feature = "matrix", feature = "value_type"))]
pub fn broadcast_array_matrix_add(
    l: Arc<crate::Array>,
    r: Arc<crate::Matrix>,
) -> Result<crate::enums::value::Value, MinarrowError> {
    let _ = (l, r);
    unimplemented!("Array-matrix broadcasting not yet implemented")
}
