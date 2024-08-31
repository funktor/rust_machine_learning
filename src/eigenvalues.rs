#![allow(dead_code)]
use crate::copy::copy;
use crate::transpose::transpose;
use crate::solve_linear::solve;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use crate::qr_decomposition::qr_decomposition;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn eigenvalues(a:&[f64], n:usize) -> Vec<f64> {
    let mut b = a.to_owned();
    let mut eig = vec![0.0;n];

    loop {
        let qr = qr_decomposition(&b, n, n);
        let x = matrix_multiply_simd(&qr.1, &qr.0, n, n, n);
        copy(&x, &mut b, n*n);

        let mut flag = true;

        for i in 0..n {
            if (b[i*n+i]-eig[i])/b[i*n+i] > 0.001 {
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

pub fn eigenvectors(a:&[f64], eigvals:&[f64], n:usize)  -> Vec<f64> {
    let mut eigvec = vec![0.0;n*n];

    for j in 0..n {
        let mut b = vec![0.0;n*n];
        copy(&a, &mut b, n*n);
        
        for i in 0..n {
            b[i*n+i] -= eigvals[j];
        }

        let y = solve(&b, &vec![0.0;n], n);
        copy(&y, &mut eigvec[j*n..(j+1)*n], n);
    }

    return eigvec;
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

    let b = matrix_multiply_simd(&a, &transpose(&a, n, m), n, m, n);
    let eig = eigenvalues(&b, n);
    let vecs = eigenvectors(&b, &eig, n);

    println!("{:?}", b);
    println!("{:?}", eig);
    println!("{:?}", vecs);
}
