#![allow(dead_code)]
use crate::row_echelon::swap_rows;
use crate::row_echelon::reduce_row;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use std::f64::MIN;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn lu_decomposition(eye:&mut Vec<f64>, l:&mut Vec<f64>, u:&mut Vec<f64>, n:usize, m:usize) {
    for j in 0..m {
        let mut mmax:f64 = MIN;
        let mut mmax_i:usize = j;

        for i in j..n {
            if u[i*m+j] > mmax {
                mmax = u[i*m+j];
                mmax_i = i;
            }
        }

        if mmax_i != j {
            swap_rows(u, m, j, mmax_i);
            swap_rows(eye, n, j, mmax_i);
            swap_rows(l, n, j, mmax_i);
        }

        for i in j+1..n {
            if u[i*m + j] != 0.0 {
                let h = u[i*m+j]/u[j*m+j];
                reduce_row(u, m, j, i, h);
                l[i*n+j] = h;
            }
        }
    }

    for i in 0..n {
        l[i*n+i] = 1.0;
    }
}

pub fn run() {
    let n = 1234;
    let m = 569;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];
    let mut u:Vec<f64> = vec![0.0;n*m];
    let mut l:Vec<f64> = vec![0.0;n*n];
    let mut eye:Vec<f64> = vec![0.0;n*n];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
        u[i] = a[i];
    }

    for i in 0..n {
        eye[i*n+i] = 1.0;
    }

    lu_decomposition(&mut eye, &mut l, &mut u, n, m);

    let x = matrix_multiply_simd(&eye, &a, n, n, m);
    let y = matrix_multiply_simd(&l, &u, n, n, m);

    for i in 0..n*m {
        if (x[i]-y[i]).abs()/x[i] > 0.01  {
            println!("{:?}, {:?}, {:?}", i, x[i], y[i]);
        }
    }

}
