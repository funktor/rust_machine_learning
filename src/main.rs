#![allow(unused_attributes)]
#![feature(portable_simd)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
mod matrix_utils;
mod orthogonal_matrices;
mod row_echelon;
mod reduced_row_echelon;
mod lu_decomposition;
mod solve_linear;
mod linear_regression_gd;
mod matrix_inverse;
mod qr_decomposition;
mod eigenvalues;
mod sparse_matrix;
mod svd;

fn main() {
    solve_linear::run();
}
