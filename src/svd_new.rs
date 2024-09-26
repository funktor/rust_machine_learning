#![allow(dead_code)]
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use crate::eigenvalues::eigenvectors_opt;
use crate::qr_decomposition::qr_decomposition_householder;
use crate::sparse_matrix::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;

fn sgn(x:f64) -> f64 {
    if x < 0.0 {
        return -1.0;
    }
    return 1.0;
}

pub fn householder_reflection_qr(a:&SparseMatrix, n:usize, m:usize) -> (SparseMatrix, SparseMatrix) {
    let mut q_lt = identity(n);
    let mut r = a.clone();

    for i in 0..min(n, m) {
        let n1 = n-i;
        let x = get_sub_mat(&r, i, n-1, i, i);
        let mut b = vec![0.0;n1];
        b[0] = 1.0;
        let e = SparseMatrix::create(n1, 1, &b);
        let x_norm = norm(&x);
        let alpha = -sgn(loc(&x, 0, 0).unwrap())*x_norm;
        let v = add(&x, &mul_const(&e, alpha));
        let z = norm(&v);
        if z > 0.0 {
            let u = mul_const(&v, 1.0/z);
            let mut w = get_sub_mat(&r, i, n-1, i, m-1);
            let mut w1 = dot(&transpose(&u), &w); 
            w1 = dot(&u, &w1);
            w1 = mul_const(&w1, 2.0);
            w = sub(&w, &w1);
            r = copy(&r, &w, i, i);
            let u1 = vstack(&SparseMatrix::create(n-n1, 1, &vec![0.0;n-n1]), &u);
            let mut q1 = dot(&transpose(&u1), &q_lt);
            q1 = dot(&u1, &q1);
            q1 = mul_const(&q1, 2.0);
            q_lt = sub(&q_lt, &q1);
        }
    }

    return (q_lt, r);
}

pub fn qr(a:&SparseMatrix, n:usize, m:usize) -> (SparseMatrix, SparseMatrix) {
    let qrd = householder_reflection_qr(&a, n, m);
    if m < n {
        let x = get_sub_mat(&transpose(&qrd.0), 0, n-1, 0, m-1);
        let y = get_sub_mat(&qrd.1, 0, m-1, 0, m-1);
        return (x, y);
    }

    return (transpose(&qrd.0), qrd.1);
}

pub fn run() {
    let n = 5;
    let m = 5;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    let a_s = SparseMatrix::create(n, m, &a);
    let b = qr(&a_s, n, m);
    let c = dot(&b.0, &b.1);

    println!("{:?}", a);
    println!();
    println!("{:?}", convert_to_array(&b.1));
    println!();
    println!("{:?}", convert_to_array(&c));
}



