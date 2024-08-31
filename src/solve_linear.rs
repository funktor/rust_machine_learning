#![allow(dead_code)]
use crate::dot_product_simd::dot_product_simd;
use crate::lu_decomposition::lu_decomposition;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn forward_sub(l:&[f64], b:&[f64], n:usize) -> Vec<f64> {
    let mut x:Vec<f64> = vec![0.0;n];

    for i in 0..n {
        let p = dot_product_simd(&l[i*n..i*(n+1)], &x[0..i]);
        x[i] = (b[i]-p)/l[i*(n+1)];
    }

    return x;
}

pub fn backward_sub(u:&[f64], b:&[f64], n:usize) -> Vec<f64> {
    let mut x:Vec<f64> = vec![0.0;n];
    
    for i in (0..n).rev() {
        let p = dot_product_simd(&u[i*(n+1)+1..(i+1)*n], &x[i+1..n]);
        x[i] = (b[i]-p)/u[i*(n+1)];
    }

    return x;
}

pub fn solve(a:&[f64], b:&[f64], n:usize) -> Vec<f64> {
    let lu = lu_decomposition(&a, n, n);

    let eye = lu.0;
    let l = lu.1;
    let u = lu.2;

    let b1 = matrix_multiply_simd(&eye, &b, n, n, 1);
    let y = forward_sub(&l, &b1, n);
    let x = backward_sub(&u, &y, n);

    return x;
}

pub fn run() {
    let n = 145;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*n];
    let mut b:Vec<f64> = vec![0.0;n];

    for i in 0..n*n {
        a[i] = normal.sample(&mut rng);
    }

    for i in 0..n {
        b[i] = normal.sample(&mut rng);
    }

    let x = solve(&a, &b, n);
    let b1 = matrix_multiply_simd(&a, &x, n, n, 1);

    for i in 0..n {
        if (b[i]-b1[i]).abs()/b[i] > 0.01  {
            println!("{:?}, {:?}, {:?}", i, b[i], b1[i]);
        }
    }
}
