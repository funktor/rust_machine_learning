#![allow(unused_attributes)]
#![feature(portable_simd)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
mod copy;
mod transpose;
// mod dot_product_simd;
mod matrix_multiplication_simd;
// mod row_echelon;
// mod lu_decomposition;
// mod solve_linear;
// mod linear_regression_gd;
// mod matrix_inverse;
// mod qr_decomposition;
// mod eigenvalues;
// mod svd;
// mod sparse_matrix;
mod svd_new;

fn main() {
    svd_new::run();
}
