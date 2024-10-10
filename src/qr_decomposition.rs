#![allow(dead_code)]
use crate::matrix_utils::*;
use crate::orthogonal_matrices::*;

pub fn givens_rotation_qr(a:&[f64], n:usize, m:usize,) -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    if n < m {
        let mut r = transpose(&a, n, m);
        let mut q = identity(m);

        for j in 0..n {
            for i in (j+1..m).rev() {
                let b = givens_left_rotation(&r, m, n, i, j, false);
                givens_left_rotation_multiply(&mut r, m, n, b.0, b.1, i, j, 0, m-1, 0, n-1);
                givens_left_rotation_multiply(&mut q, m, m, b.0, b.1, i, j, 0, m-1, 0, m-1);
            }
        }

        return (transpose(&r, m, n), q, n, m, m);
    }
    else {
        let mut r = a.to_vec();
        let mut q = identity(n);

        for j in 0..m {
            for i in (j+1..n).rev() {
                let b = givens_left_rotation(&r, n, m, i, j, false);
                givens_left_rotation_multiply(&mut r, n, m, b.0, b.1, i, j, 0, n-1, 0, m-1);
                givens_left_rotation_multiply(&mut q, n, n, b.0, b.1, i, j, 0, n-1, 0, n-1);
            }
        }

        return (transpose(&q, n, n), r, n, n, m);
    }
    
}

pub fn householder_reflection_qr(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    if n < m {
        let a1 = transpose(&a, n, m);
        let (q, r) = householder_reflection_left_multiply(&a1, m, n);
        return (transpose(&r, m, n), q, n, m, m);
    }
    else {
        let (q, r) = householder_reflection_left_multiply(&a, n, m);
        return (transpose(&q, n, n), r, n, n, m);
    }
}