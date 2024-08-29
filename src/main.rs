#![allow(unused_attributes)]
#![feature(portable_simd)]
mod dot_product_simd;
mod matrix_multiplication_simd;
mod row_echelon;
mod lu_decomposition;
mod solve_linear;
mod linear_regression_gd;
mod matrix_inverse;

fn main() {
    matrix_inverse::run();
}
