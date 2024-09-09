#![allow(dead_code)]
use crate::copy::copy;
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use crate::qr_decomposition::qr_decomposition_householder;
use crate::qr_decomposition::norm;
use crate::qr_decomposition::add_vecs;
use crate::qr_decomposition::mul_vec;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn eigenvalues(a:&[f64], n:usize) -> Vec<f64> {
    let mut b = a.to_owned();
    let mut eig = vec![0.0;n];

    loop {
        let s = b[n*n-1];
        for i in 0..n {
            b[i*n+i] -= s;
        }

        let qr = qr_decomposition_householder(&b, n, n);
        let x = matrix_multiply_simd(&qr.1, &qr.0, n, n, n);
        copy(&x, &mut b, n*n);

        let mut flag = true;

        for i in 0..n {
            b[i*n+i] += s;
            if (b[i*n+i]-eig[i]).abs()/b[i*n+i] > 0.001 {
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

    let mut b = a.to_owned();

    for i in 0..n {
        eigvec[i*n+i] = 1.0;
    }

    loop {
        let qr = qr_decomposition_householder(&b, n, n);
        eigvec = matrix_multiply_simd(&eigvec, &qr.0, n, n, n);

        let x = matrix_multiply_simd(&qr.1, &qr.0, n, n, n);
        copy(&x, &mut b, n*n);

        let mut flag = true;

        for i in 0..n {
            if (b[i*n+i]-eigval[i]).abs()/b[i*n+i] > 0.001 {
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

pub fn eigenvectors_opt(a:&[f64], n:usize)  -> (Vec<f64>, Vec<f64>) {
    let mut eigvec = vec![0.0;n*n];
    let mut eigval = vec![0.0;n];

    let b = a.to_owned();
    let mut r = transpose(&b, n, n);

    for i in 0..n {
        eigvec[i*n+i] = 1.0;
    }

    loop {
        let mut q:Vec<f64> = vec![0.0;n*n];

        for i in 0..n {
            q[i*n+i] = 1.0;
        }

        for i in 0..n-1 {
            let n1 = n-i;
            let a1 = &r[i*(n+1)..(i+1)*n];
            let x = norm(&a1, n1);
    
            let mut alpha;
        
            if a1[0] < 0.0 {
                alpha = -1.0;
            }
            else {
                alpha = 1.0;
            }
    
            alpha = -alpha*x;
    
            let mut w = vec![0.0;n1];
            w[0] = alpha;
            let v = add_vecs(&a1, &w, n1);
            let y = norm(&v, n1);
            let u = mul_vec(&v, 1.0/y, n1);    
    
            let mut u1 = vec![0.0;n];
            copy(&u, &mut u1[i..n], n1);
    
            let mut z = matrix_multiply_simd(&r, &u1, n, n, 1);
            z = matrix_multiply_simd(&z, &u1, n, 1, n);
            z = mul_vec(&z, -2.0, n*n);
            r = add_vecs(&r, &z, n*n);
    
            z = matrix_multiply_simd(&q, &u1, n, n, 1);
            z = matrix_multiply_simd(&z, &u1, n, 1, n);
            z = mul_vec(&z, -2.0, n*n);
            q = add_vecs(&q, &z, n*n);

            z = matrix_multiply_simd(&eigvec, &u1, n, n, 1);
            z = matrix_multiply_simd(&z, &u1, n, 1, n);
            z = mul_vec(&z, -2.0, n*n);
            eigvec = add_vecs(&eigvec, &z, n*n);
        }

        let x = matrix_multiply_simd(&transpose(&q, n, n), &r, n, n, n);
        copy(&x, &mut r, n*n);

        let mut flag = true;

        for i in 0..n {
            if (r[i*n+i]-eigval[i]).abs()/r[i*n+i] > 0.001 {
                flag = false;
            }
            eigval[i] = r[i*n+i];
        }

        if flag {
            break;
        }
    }

    return (eigval, eigvec);
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
    let vecs = eigenvectors_opt(&b, n);

    println!("{:?}", b);
    println!("{:?}", eig);
    println!("{:?}", vecs.0);
    println!("{:?}", vecs.1);
}
