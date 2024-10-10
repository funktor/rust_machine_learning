#![allow(dead_code)]
use crate::matrix_utils::*;
use crate::orthogonal_matrices::*;
use crate::qr_decomposition::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::min;
use std::time::SystemTime;

pub fn golub_kahan(a:&mut [f64], l:&mut [f64], r:&mut [f64], n:usize, m:usize, z:usize, i:usize, j:usize) {
    let mu = eigenvalue_bidiagonal_slices(&a, z, z, i, j, i, j);
    
    let u = a[i*z+i];
    let v = a[i*z+i+1];
    
    a[i*z+i] = u*u-mu;
    a[i*z+i+1] = u*v;
    
    for k in i..j {
        let mut x;
        let mut y;

        if k > i {
            x = k-1;
            y = k+1; 
        }
        else {
            x = i;
            y = i+1; 
        }

        let b = givens_right_rotation(&a, z, z, x, y, false);

        if k == i {
            a[i*z+i] = u;
            a[i*z+i+1] = v;
        }

        givens_right_rotation_multiply(a, z, z, b.0, b.1, x, y, i, j, i, j);
        givens_right_rotation_multiply(r, m, z, b.0, b.1, x, y, 0, m-1, 0, z-1);

        if k > i {
            x = k+1;
            y = k;
        }
        else {
            x = i+1;
            y = i;
        }
            
        let b = givens_left_rotation(&a, z, z, x, y, false);

        givens_left_rotation_multiply(a, z, z, b.0, b.1, x, y, i, j, i, j);
        givens_left_rotation_multiply(l, z, n, b.0, b.1, x, y, 0, z-1, 0, n-1);
    }
}

pub fn golub_reisch_svd(a:&[f64], mut n:usize, mut m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a1 = a.to_vec();
    let mut do_transpose = false;

    if n < m {
        do_transpose = true;
        a1 = transpose(&a1, n, m);
        let g = n;
        n = m;
        m = g;
    }
    
    let hr = householder_reflection_bidiagonalization(&a1, n, m);
    let r = min(n, m);

    let mut u = sub_mat(&hr.0, n, n, 0, r-1, 0, n-1);
    a1 = sub_mat(&hr.1, n, m, 0, r-1, 0, r-1);
    let mut v = sub_mat(&hr.2, m, m, 0, m-1, 0, r-1);
    
    let eps = 1e-7;
    
    loop {
        for i in 0..r-1 {
            if a1[i*r+i+1].abs() < eps*(a1[i*r+i].abs() + a1[(i+1)*r+i+1].abs()) {
                a1[i*r+i+1] = 0.0;
            }
        }

        let mut q = 0;
        for i in (0..r-1).rev() {
            if a1[i*r+i+1].abs() > 0.0 {
                q = i+1;
                break;
            }
        }

        if q == 0 {
            break;
        }

        let mut p = 0;
        for i in (0..q).rev() {
            if a1[i*r+i+1].abs() == 0.0 {
                p = i+1;
                break;
            }
        }

        let mut flag: bool = false;
        
        for i in p..q {
            if a1[i*r+i].abs() == 0.0 {
                flag = true;
                for j in i+1..r {
                    let b = givens_left_rotation(&a1, r, r, i+1, j, true);
                    
                    givens_left_rotation_multiply(&mut a1, r, r, b.0, b.1, i+1, j, 0, r-1, 0, r-1);
                    givens_left_rotation_multiply(&mut u, r, n, b.0, b.1, i+1, j, 0, r-1, 0, n-1);
                }
            }
        }

        if !flag && p < q {
            golub_kahan(&mut a1, &mut u, &mut v, n, m, r, p, q);
        }
    }

    if do_transpose {
        return (v, transpose(&a1, r, r), u);
    }
        
    return (transpose(&u, r, n), a1, transpose(&v, m, r));
}

pub fn randomized_svd(a:&[f64], n:usize, m:usize, k:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>){
    let l = min(k+10, m);
    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut p:Vec<f64> = vec![0.0;m*l];

    for i in 0..m*l {
        p[i] = normal.sample(&mut rng);
    }

    let b = matrix_multiply_simd(&a, &p, n, m, l);
    let (mut q, _r, n1, m1, _p1) = givens_rotation_qr(&b, n, l);
    let w= min(m1, l);
    q = sub_mat(&q, n1, m1, 0, n1-1, 0, w-1);

    let c = &matrix_multiply_simd(&transpose(&q, n1, w), &a, w, n1, m);
    let (mut u, s, v) = golub_reisch_svd(&c, w, m);
    u = matrix_multiply_simd(&q, &u, n1, w, w);
    return (u, s, v);
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

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    let b = golub_reisch_svd(&a, n, m);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    println!("{:?}", end_time-start_time);
    let r = min(n, m);
    let mut c = matrix_multiply_simd(&b.0, &b.1, n, r, r);
    c = matrix_multiply_simd(&c, &b.2, n, r, m);

    // println!("{:?}", a);
    // println!();
    // println!("{:?}", b.1);
    // println!();
    // println!("{:?}", c);

    for i in 0..n*m {
        assert!((c[i]-a[i]).abs() < 1e-5, "Some issue in SVD !!! {}, {}", a[i], c[i]);
    }
}



