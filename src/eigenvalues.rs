#![allow(dead_code)]
use crate::qr_decomposition::*;
use crate::matrix_utils::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn eigenvalues(a:&[f64], n:usize) -> Vec<f64> {
    let mut b = a.to_vec();
    let mut eig = vec![0.0;n];

    loop {
        let s = b[n*n-1];
        for i in 0..n {
            b[i*n+i] -= s;
        }

        let qr = householder_reflection_qr(&b, n, n);
        b = matrix_multiply_simd(&qr.1, &qr.0, n, n, n);

        let mut flag = true;

        for i in 0..n {
            b[i*n+i] += s;
            if (b[i*n+i]-eig[i]).abs()/b[i*n+i].abs() > 0.001 {
                flag = false;
            }
            eig[i] = b[i*n+i];
        }

        if flag {
            break;
        }
    }

    return eig;
}

pub fn eigenvectors(a:&[f64], n:usize)  -> (Vec<f64>, Vec<f64>) {
    let mut eigvec = vec![0.0;n*n];
    let mut eigval = vec![0.0;n];

    let mut b = a.to_vec();

    for i in 0..n {
        eigvec[i*n+i] = 1.0;
    }

    loop {
        let qr = householder_reflection_qr(&b, n, n);
        eigvec = matrix_multiply_simd(&eigvec, &qr.0, n, n, n);

        b = matrix_multiply_simd(&qr.1, &qr.0, n, n, n);

        let mut flag = true;

        for i in 0..n {
            if (b[i*n+i]-eigval[i]).abs()/b[i*n+i].abs() > 0.001 {
                flag = false;
            }
            eigval[i] = b[i*n+i];
        }

        if flag {
            break;
        }
    }

    return (eigval, eigvec);
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

    let b = matrix_multiply_simd(&a, &transpose(&a, n, m), n, m, n);
    let eig = eigenvalues(&b, n);
    let vecs = eigenvectors(&b, n);

    println!("{:?}", b);
    println!();
    println!("{:?}", eig);
    println!();
    println!("{:?}", vecs.0);
    println!();
    println!("{:?}", vecs.1);
}
