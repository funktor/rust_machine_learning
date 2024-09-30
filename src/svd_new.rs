#![allow(dead_code)]
use crate::sparse_matrix::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;
use std::time::SystemTime;

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
            r = copy(&r, &w, i, n-1, i, m-1);

            let u1 = vstack(&SparseMatrix::create(n-n1, 1, &vec![0.0;n-n1]), &u);
            let mut q1 = dot(&transpose(&u1), &q_lt);
            q1 = dot(&u1, &q1);
            q1 = mul_const(&q1, 2.0);
            q_lt = sub(&q_lt, &q1);
        }
    }

    return (q_lt, r);
}

pub fn givens_right_rotation(a:&SparseMatrix, i:usize, j:usize) -> (f64, f64) {
    let x = loc(&a, i, j-1).unwrap();
    let y = loc(&a, i, j).unwrap();
    let r = (x*x + y*y).sqrt();

    return (x/r, -y/r);
}

pub fn givens_right_rotation_multiply(a:&SparseMatrix, c:f64, s:f64, _i:usize, j:usize) -> SparseMatrix{
    let n = a.nrow;
    let b = SparseMatrix::create(2, 2, &vec![c, s, -s, c]);
    let d = dot(&get_sub_mat(&a, 0, n-1, j-1, j), &b);
    let a1 = copy(&a, &d, 0, n-1, j-1, j);
    return a1;
}

pub fn givens_left_rotation(a:&SparseMatrix, i:usize, j:usize) -> (f64, f64) {
    let x = loc(&a, i-1, j).unwrap();
    let y = loc(&a, i, j).unwrap();
    let r = (x*x + y*y).sqrt();

    return (x/r, -y/r);
}

pub fn givens_left_rotation_multiply(a:&SparseMatrix, c:f64, s:f64, i:usize, _j:usize) -> SparseMatrix {
    let m = a.ncol;
    let b = SparseMatrix::create(2, 2, &vec![c, -s, s, c]);
    let d = dot(&b, &get_sub_mat(&a, i-1, i, 0, m-1));
    let a1 = copy(&a, &d, i-1, i, 0, m-1);
    return a1;
}

pub fn givens_rotation_qr(a:&SparseMatrix) -> (SparseMatrix, SparseMatrix) {
    let n = a.nrow;
    let m = a.ncol;
    let mut q = identity(n);
    let mut r = a.clone();
    
    for j in 0..m {
        for i in (j+1..n).rev() {
            let b = givens_left_rotation(&r, i, j);
            r = givens_left_rotation_multiply(&r, b.0, b.1, i, j);
            q = givens_left_rotation_multiply(&q, b.0, b.1, i, j);
        }
    }

    return (q, r);
}

pub fn qr(a:&SparseMatrix, n:usize, m:usize) -> (SparseMatrix, SparseMatrix) {
    let qrd = givens_rotation_qr(&a);
    if m < n {
        let x = get_sub_mat(&transpose(&qrd.0), 0, n-1, 0, m-1);
        let y = get_sub_mat(&qrd.1, 0, m-1, 0, m-1);
        return (x, y);
    }

    return (transpose(&qrd.0), qrd.1);
}

pub fn run() {
    let n = 500;
    let m = 500;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    let a_s = SparseMatrix::create(n, m, &a);
    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    let b = qr(&a_s, n, m);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    println!("{:?}", end_time-start_time);
    // let c = dot(&b.0, &b.1);


    // println!("{:?}", a);
    // println!();
    // println!("{:?}", convert_to_array(&b.1));
    // println!();
    // println!("{:?}", convert_to_array(&c));
}



