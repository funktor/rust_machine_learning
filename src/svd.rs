#![allow(dead_code)]
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use crate::eigenvalues::eigenvectors;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;

pub fn svd(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r = min(n, m);
    let a_t = transpose(&a, n, m);

    let b = matrix_multiply_simd(&a, &a_t, n, m, n);
    let c = matrix_multiply_simd(&a_t, &a, m, n, m);
    let b_eig = eigenvectors(&b, n);
    let c_eig = eigenvectors(&c, m);

    let eigvals = b_eig.0;
    let mut s = vec![0.0;n*m];

    for i in 0..r {
        s[i*m+i] = eigvals[i].sqrt();
    }

    let u = b_eig.1;
    let v = c_eig.1;

    return (u, s, transpose(&v, m, m));
}

pub fn run() {
    let n = 3;
    let m = 3;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    let res = svd(&a, n, m);

    println!("{:?}", a);
    println!("{:?}", res.0);
    println!("{:?}", res.1);
    println!("{:?}", res.2);

    let p = matrix_multiply_simd(&res.0, &res.1, n, n, m);
    let q = matrix_multiply_simd(&p, &res.2, n, m, m);

    println!("{:?}", q);
}
