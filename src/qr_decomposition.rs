#![allow(dead_code)]
use crate::copy::copy;
use crate::dot_product_simd::dot_product_simd;
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::matrix_multiply_simd;
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;

pub fn norm(a:&[f64], n:usize) -> f64{
    const LANES:usize = 64;
    let mut s = 0.0;

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                s += a[j]*a[j];
            }
        }
        else {
            let x:Simd<f64, LANES> = Simd::from_slice(&a[i..i+LANES]);
            let y = x*x;
            s += y.reduce_sum();
        }
    }

    return s.sqrt();
}

pub fn mul_vec(inp:&[f64], x:f64, n:usize) -> Vec<f64>{
    const LANES:usize = 64;

    let mut out = vec![0.0;n];
    let b = Simd::splat(x);

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                out[j] = inp[j]*x;
            }
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp[i..i+LANES]);
            let y = a*b;
            Simd::copy_to_slice(y, &mut out[i..i+LANES]);
        }
    }

    return out;
}

pub fn mul_mat_const(inp:&[f64], x:f64, n:usize, m:usize) -> Vec<f64>{
    let mut out = vec![0.0;n*m];
    
    for i in 0..n {
        let a = mul_vec(&inp[i*m..(i+1)*m], x, m);
        copy(&a, &mut out[i*m..(i+1)*m], m);
    }

    return out;
}

pub fn add_vecs(inp1:&[f64], inp2:&[f64], n:usize) -> Vec<f64>{
    const LANES:usize = 64;
    let mut out = vec![0.0;n];

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                out[j] = inp1[j] + inp2[j];
            }
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp1[i..i+LANES]);
            let b:Simd<f64, LANES> = Simd::from_slice(&inp2[i..i+LANES]);

            let y = a + b;
            Simd::copy_to_slice(y, &mut out[i..i+LANES]);
        }
    }

    return out;
}

pub fn sub_vecs(inp1:&[f64], inp2:&[f64], n:usize) -> Vec<f64>{
    const LANES:usize = 64;
    let mut out = vec![0.0;n];

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                out[j] = inp1[j] - inp2[j];
            }
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp1[i..i+LANES]);
            let b:Simd<f64, LANES> = Simd::from_slice(&inp2[i..i+LANES]);

            let y = a - b;
            Simd::copy_to_slice(y, &mut out[i..i+LANES]);
        }
    }

    return out;
}

pub fn qr_decomposition(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let a_t = transpose(&a, n, m);

    let mut q:Vec<f64> = vec![0.0;n*n];
    let mut r:Vec<f64> = vec![0.0;n*m];

    for i in 0..m {
        let b = &a_t[i*n..(i+1)*n];

        if i < n {
            let mut s = vec![0.0;n];
            for j in 0..i {
                let q1 = &q[j*n..(j+1)*n];
                let p = dot_product_simd(b, q1);
                r[j*m+i] = p;
                let c = mul_vec(q1, p, n);
                s = add_vecs(&s, &c, n);
            }

            let u = sub_vecs(&b, &s, n);
            let h = norm(&u, n);
            let w = mul_vec(&u, 1.0/h, n);
            copy(&w, &mut q[i*n..(i+1)*n], n);
            r[i*m+i] = h;
        }
        else {
            for j in 0..n {
                let q1 = &q[j*n..(j+1)*n];
                let p = dot_product_simd(b, q1);
                r[j*m+i] = p;
            }
        }
    }

    let mut q_t = transpose(&q, n, n);
    let mut sign_r = vec![0.0;n*n];

    for i in 0..n {
        if r[i*m+i] < 0.0 {
            sign_r[i*m+i] = -1.0;
        }
        else if r[i*m+i] > 0.0 {
            sign_r[i*m+i] = 1.0;
        }
    }

    q_t = matrix_multiply_simd(&q_t, &sign_r, n, n, n);
    r = matrix_multiply_simd(&sign_r, &r, n, n, m);

    return (q_t, r);
}

pub fn qr_decomposition_householder(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let a_t = transpose(&a, n, m);

    let mut q:Vec<f64> = vec![0.0;n*n];
    let mut r:Vec<f64> = vec![0.0;n*m];

    copy(&a_t, &mut r, n*m);

    for i in 0..n {
        q[i*n+i] = 1.0;
    }

    let mut q_matrices: Vec<Vec<f64>> = Vec::new();

    for i in 0..min(n,m)-1 {
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
        let u_t = transpose(&u, n1, 1);

        let mut u1 = vec![0.0;n];
        for j in i..n {
            u1[j] = u[j-i];
        }
        let u1_t = transpose(&u1, n, 1);

        let k = m-i;
        let mut z = matrix_multiply_simd(&r[i*n..], &u1, k, n, 1);
        z = matrix_multiply_simd(&z, &u1_t, k, 1, n);
        z = mul_mat_const(&z, -2.0, k, n);

        for j in i..m {
            let b = add_vecs(&r[j*n..(j+1)*n], &z[(j-i)*n..(j-i+1)*n], n);
            copy(&b, &mut r[j*n..(j+1)*n], n);
        }

        

        let mut g = matrix_multiply_simd(&u, &u_t, n1, 1, n1);
        g = mul_mat_const(&g, -2.0, n1, n1);

        for j in 0..n1 {
            g[j*n1+j] = 1.0+g[j*n1+j];
        }

        let mut q1 = vec![0.0;n*n];
        for j in 0..n {
            if j < i {
                q1[j*n+j] = 1.0;
            }
            else {
                copy(&g[(j-i)*n1..(j-i+1)*n1], &mut q1[j*n+i..(j+1)*n], n1);
            }
        }

        let q1_t = transpose(&q1, n, n);
        q_matrices.push(q1_t);
    }   

    for q_mat in q_matrices {
        q = matrix_multiply_simd(&q, &q_mat, n, n, n);
    }

    return (q, transpose(&r, m, n));
}


pub fn run() {
    let n = 1234;
    let m = 569;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    let qr = qr_decomposition_householder(&a, n, m);
    let v = matrix_multiply_simd(&qr.0, &qr.1, n, n, m);

    // println!("{:?}", a);
    // println!("{:?}", qr.0);
    // println!("{:?}", qr.1);
    // println!("{:?}", v);

    for i in 0..n*m {
        if (a[i]-v[i]).abs()/a[i] > 0.01  {
            println!("{:?}, {:?}, {:?}", i, a[i], v[i]);
        }
    }
}
