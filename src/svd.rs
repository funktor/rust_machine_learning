#![allow(dead_code)]
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use crate::eigenvalues::eigenvectors_opt;
use crate::qr_decomposition::qr_decomposition_householder;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;

pub fn svd(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r = min(n, m);
    let a_t = transpose(&a, n, m);

    let b = matrix_multiply_simd(&a, &a_t, n, m, n);
    let b_eig = eigenvectors_opt(&b, n);

    let eigvals = b_eig.0;
    let mut s = vec![0.0;n*m];

    for i in 0..r {
        s[i*m+i] = eigvals[i].sqrt();
    }

    let u = b_eig.1;
    let v1 = matrix_multiply_simd(&a_t, &u, m, n, n);

    let mut s_inv = vec![0.0;n*m];

    for i in 0..r {
        s_inv[i*m+i] = 1.0/s[i*m+i];
    }

    let v = matrix_multiply_simd(&v1, &s_inv, m, n, m);

    return (u, s, transpose(&v, m, m));
}

pub fn svd_low_rank(a:&[f64], n:usize, m:usize, rank:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r = min(rank, min(n, m));
    let a_t = transpose(&a, n, m);

    let b = matrix_multiply_simd(&a, &a_t, n, m, n);
    let b_eig = eigenvectors_opt(&b, n);

    let eigvals = b_eig.0;
    let mut s = vec![0.0;r*r];

    for i in 0..r {
        s[i*r+i] = eigvals[i].sqrt();
    }

    let mut u = b_eig.1;
    let u_t = transpose(&u, n, n);
    u = transpose(&u_t[..r*n], r, n);

    let v1 = matrix_multiply_simd(&a_t, &u, m, n, r);
    let mut s_inv = vec![0.0;r*r];

    for i in 0..r {
        s_inv[i*r+i] = 1.0/s[i*r+i];
    }

    let v = matrix_multiply_simd(&v1, &s_inv, m, r, r);
    return (u, s, transpose(&v, m, r));
}

pub fn svd_randomized(a:&[f64], n:usize, m:usize, k:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut p:Vec<f64> = vec![0.0;m*k];

    for i in 0..m*k {
        p[i] = normal.sample(&mut rng);
    }

    let z = matrix_multiply_simd(&a, &p, n, m, k);
    let qr = qr_decomposition_householder(&z, n, k);
    let q = qr.0;
    let y = matrix_multiply_simd(&transpose(&q, n, n), &a, n, n, m);
    let r_svd = svd_low_rank(&y, n, m, k);
    let u = matrix_multiply_simd(&q, &r_svd.0, n, n, k);
    
    return (u, r_svd.1, r_svd.2);
}

pub fn run() {
    let n = 500;
    let m = 500;
    let k = 30;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    let res = svd_randomized(&a, n, m, k);

    let p = matrix_multiply_simd(&res.0, &res.1, n, k, k);
    let q = matrix_multiply_simd(&p, &res.2, n, k, m);

    println!("{:?}", &a[..30]);
    println!();
    println!("{:?}", &q[..30]);

    // for i in 0..n*m {
    //     if (a[i]-q[i]).abs()/a[i] > 0.01  {
    //         println!("{:?}, {:?}, {:?}", i, a[i], q[i]);
    //     }
    // }
}
