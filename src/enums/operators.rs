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

//! Contains basic numeric kernel operators for matching and routing purposes

/// Arithmetic operators for numeric computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOperator {
    /// Addition (`lhs + rhs`)
    Add,
    /// Subtraction (`lhs - rhs`)
    Subtract,
    /// Multiplication (`lhs * rhs`)
    Multiply,
    /// Division (`lhs / rhs`)
    ///
    /// Division is true division: `7 / 2` is `3.5`. The `Scalar` arms
    /// return a `Float64` scalar for integer operands, and callers dividing
    /// integer arrays cast their operands to `f64` before dispatch so the
    /// float kernels produce the float result.
    ///
    /// The integer slice kernels themselves serve `FloorDiv`, so `Divide`
    /// handed raw integer slices at the kernel level behaves as floor
    /// division. Cast to float first for a true-division result.
    ///
    /// Division by zero on floats follows IEEE 754: a nonzero value over
    /// zero yields Inf with the operands' sign, and zero over zero yields
    /// NaN. On raw integer slices it panics in unmasked arrays and
    /// nullifies in masked arrays.
    Divide,
    /// Modulus/remainder operation (`lhs % rhs`)
    ///
    /// Behaviour matches Rust's `%` operator: the result keeps the
    /// dividend's sign, so `-7 % 2` is `-1`. Division by zero handling
    /// follows same rules as `Divide` operation.
    Remainder,
    /// Exponentiation (`lhs ^ rhs`)
    ///
    /// For integers, exponentiation by squaring with wrapping
    /// multiplication, so overflow wraps like the other integer arms. The
    /// exponent must convert to `u32`: a negative or larger exponent
    /// returns an error advising a cast to float. For floating-point, uses
    /// logarithmic computation.
    Power,
    /// Floor division (`lhs // rhs`)
    ///
    /// Rounds the quotient towards negative infinity. For unsigned integers this is
    /// identical to truncation division. For signed integers, when the remainder is
    /// non-zero and the operands have different signs, the result is one less than
    /// truncation division. For floating-point, equivalent to `(lhs / rhs).floor()`.
    FloorDiv,
}

/// Comparison operators for binary predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    /// Equality comparison (`lhs == rhs`)
    Equals,
    /// Inequality comparison (`lhs != rhs`)
    NotEquals,
    /// Less-than comparison (`lhs < rhs`)
    LessThan,
    /// Less-than-or-equal comparison (`lhs <= rhs`)
    LessThanOrEqualTo,
    /// Greater-than comparison (`lhs > rhs`)
    GreaterThan,
    /// Greater-than-or-equal comparison (`lhs >= rhs`)
    GreaterThanOrEqualTo,
    /// Tests if value is null (`lhs IS NULL`)
    ///
    /// Always returns a valid boolean, never null.
    IsNull,
    /// Tests if value is not null (`lhs IS NOT NULL`)
    ///
    /// Always returns a valid boolean, never null.
    IsNotNull,
    /// Range membership test (`lhs BETWEEN min AND max`)
    ///
    /// Equivalent to `lhs >= min AND lhs <= max` with appropriate null handling.
    Between,
    /// Set membership test (`lhs IN (set)`)
    ///
    /// Returns true if lhs matches any value in the provided set.
    In,
    /// Set exclusion test (`lhs NOT IN (set)`)
    ///
    /// Returns true if lhs doesn't match any value in the provided set.
    NotIn,
}

/// Logical/boolean operators for conditional expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOperator {
    /// Logical AND (`lhs AND rhs`)
    ///
    /// Returns false if either operand is false, otherwise propagates nulls.
    And,
    /// Logical OR (`lhs OR rhs`)
    ///
    /// Returns true if either operand is true, otherwise propagates nulls.
    Or,
    /// Logical XOR (`lhs XOR rhs`)
    ///
    /// Returns true if operands differ, false if same, null if either is null.
    Xor,
}

/// Bitwise operators for integer values.
///
/// These operate on the integer bit patterns element-wise. Unary bitwise
/// complement is expressed through [`UnaryOperator::Not`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseOperator {
    /// Bitwise AND (`lhs & rhs`)
    And,
    /// Bitwise OR (`lhs | rhs`)
    Or,
    /// Bitwise XOR (`lhs ^ rhs`)
    Xor,
}

/// Unary operators for single-operand transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Arithmetic negation (`-operand`)
    ///
    /// Negates numeric values. For unsigned integers, uses wrapping negation.
    Negative,
    /// Logical/bitwise NOT (`!operand` or `~operand`)
    ///
    /// For booleans: logical NOT. For integers: bitwise complement.
    Not,
    /// Unary plus (`+operand`)
    ///
    /// Identity operation that explicitly indicates positive values.
    /// Primarily used for symmetry with negation operator.
    Positive,
}
