#![allow(dead_code)]
use crate::copy::copy;
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::{matrix_multiply_simd, matrix_multiply_simd_slices};
use std::f64::consts::LOG10_2;
use std::f64::{MAX, MIN_POSITIVE};
// use crate::sparse_matrix::*;
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::cmp::{min, max};
use std::time::SystemTime;

fn sgn(x:f64) -> f64 {
    if x < 0.0 {
        return -1.0;
    }
    return 1.0;
}

fn identity(n:usize) -> Vec<f64> {
    let mut q = vec![0.0;n*n];
    for i in 0..n {
        q[i*(n+1)] = 1.0;
    }

    return q;
}

fn my_sqrt(x:f64) -> f64 {
    if x > 0.0 {
        return x.sqrt().abs();
    }
    return 0.0;
}

fn hypot(x:f64, y:f64) -> f64 {
    let mut i = 1.0;
    let mut s = 1.0;
    let mut r = 0.5;
    let w;
    let u;
    if x.abs() > y.abs() {
        w = (y/x).abs();
        u = x.abs();
    }
    else {
        w = (x/y).abs();
        u = y.abs();
    }
    let mut z = w*w;

    loop {
        let p = (r/i)*z;
        if p.abs()/s < 0.01 {
            break;
        }
        s += p;
        r *= 0.5-i;
        i *= i+1.0;
        z *= w*w;
    }

    return s.abs()*u;
}

// pub fn norm_row(a:&[f64], m:usize, i:usize, j1:usize, j2:usize) -> f64{
//     const LANES:usize = 64;
//     let mut s = 0.0;

//     for k in (i*m+j1..i*m+j2+1).step_by(LANES) {
//         if k+LANES > i*m+j2+1 {
//             for h in k..i*m+j2+1 {
//                 s += a[h]*a[h];
//             }
//         }
//         else {
//             let x:Simd<f64, LANES> = Simd::from_slice(&a[k..k+LANES]);
//             let y = x*x;
//             s += y.reduce_sum();
//         }
//     }

//     return s.sqrt();
// }

// pub fn norm_col(a:&[f64], m:usize, j:usize, i1:usize, i2:usize) -> f64{
//     const LANES:usize = 64;
//     let mut s = 0.0;

//     for k in (i*m+j1..i*m+j2+1).step_by(LANES) {
//         if k+LANES > i*m+j2+1 {
//             for h in k..i*m+j2+1 {
//                 s += a[h]*a[h];
//             }
//         }
//         else {
//             let x:Simd<f64, LANES> = Simd::from_slice(&a[k..k+LANES]);
//             let y = x*x;
//             s += y.reduce_sum();
//         }
//     }

//     return s.sqrt();
// }

pub fn sub_mat(a:&[f64], _n:usize, m:usize, r_start:usize, r_end:usize, c_start:usize, c_end:usize) -> Vec<f64> {
    let mut b:Vec<f64> = vec![0.0;(r_end-r_start+1)*(c_end-c_start+1)];
    let u = c_end-c_start+1;
    let mut k = 0;

    for i in r_start..(r_end+1) {
        b[k*u..(k+1)*u].copy_from_slice(&a[i*m+c_start..i*m+c_end+1]);
        k += 1;
    }

    return b;
}

pub fn copy_sub_mat(a:&mut [f64], b:&[f64], _n:usize, m:usize, r_start:usize, r_end:usize, c_start:usize, c_end:usize) {
    let u = c_end-c_start+1;
    let mut k = 0;

    for i in r_start..(r_end+1) {
        a[i*m+c_start..i*m+c_end+1].copy_from_slice(&b[k*u..(k+1)*u]);
        k += 1;
    }
}

pub fn add(inp1:&[f64], inp2:&[f64], n:usize) -> Vec<f64>{
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

pub fn sub(inp1:&[f64], inp2:&[f64], n:usize) -> Vec<f64>{
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

pub fn mul_const(inp:&[f64], x:f64, n:usize) -> Vec<f64>{
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

pub fn householder_reflection_left_multiply(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let mut q_lt = identity(n);
    let mut r = a.to_vec();
    let w = min(n, m);

    for i in 0..w {
        let n1 = n-i;

        let mut nm = 0.0;
        let mut u = vec![0.0;n1];

        for i1 in i..n {
            nm += r[i1*m+i]*r[i1*m+i];
            u[i1-i] = r[i1*m+i];
        }

        u[0] -= sgn(r[i*m+i])*my_sqrt(nm);
        let z = my_sqrt(nm-r[i*m+i]*r[i*m+i]+u[0]*u[0]);

        if z > 0.0 {
            for i1 in 0..n1 {
                u[i1] = u[i1]/z;
            }

            let mut r1 = vec![0.0;m-i];
            for i1 in i..n {
                for j1 in i..m {
                    r1[j1-i] += u[i1-i]*r[i1*m+j1];
                }
            }

            // let r1 = matrix_multiply_simd_slices(&u, &r, 1, n1, m, 0, 0, 0, n1-1, i, n-1, i, m-1);

            for i1 in i..n {
                for j1 in i..m {
                    r[i1*m+j1] -= 2.0*u[i1-i]*r1[j1-i];
                }
            }

            let mut q1 = vec![0.0;n];
            for i1 in i..n {
                for j1 in 0..n {
                    q1[j1] += u[i1-i]*q_lt[i1*n+j1];
                }
            }

            for i1 in i..n {
                for j1 in 0..n {
                    q_lt[i1*n+j1] -= 2.0*u[i1-i]*q1[j1];
                }
            }
        }
    }

    return (q_lt, r);
}

pub fn householder_reflection_right_multiply(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let mut q_rt = identity(m);
    let mut r = a.to_vec();

    for i in 0..min(n, m) {
        if m-i-1 > 0 {
            let n1 = m-i-1;

            let mut nm = 0.0;
            let mut u = vec![0.0;n1];

            for j1 in i+1..m {
                nm += r[i*m+j1]*r[i*m+j1];
                u[j1-i-1] = r[i*m+j1];
            }

            u[0] -= sgn(r[i*m+i+1])*my_sqrt(nm);
            let z = my_sqrt(nm-r[i*m+i+1]*r[i*m+i+1]+u[0]*u[0]);

            if z > 0.0 {
                for i1 in 0..n1 {
                    u[i1] = u[i1]/z;
                }

                let mut r1 = vec![0.0;n-i];
                for i1 in i..n {
                    for j1 in i+1..m {
                        r1[i1-i] += u[j1-i-1]*r[i1*m+j1];
                    }
                }

                // let r1 = matrix_multiply_simd_slices(&r, &u, n, m, 1, i, n-1, i+1, m-1, 0, n1-1, 0, 0);

                for i1 in i..n {
                    for j1 in i+1..m {
                        r[i1*m+j1] -= 2.0*u[j1-i-1]*r1[i1-i];
                    }
                }
                
                let mut q1 = vec![0.0;m];
                for i1 in 0..m {
                    for j1 in i+1..m {
                        q1[i1] += u[j1-i-1]*q_rt[i1*m+j1];
                    }
                }

                for i1 in 0..m {
                    for j1 in i+1..m {
                        q_rt[i1*m+j1] -= 2.0*u[j1-i-1]*q1[i1];
                    }
                }
            }
        }
    }

    return (q_rt, r);
}


pub fn householder_reflection_bidiagonalization(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut q_lt = identity(n);
    let mut q_rt = identity(m);

    let mut r = a.to_vec();

    for i in 0..min(n, m) {
        let n1 = n-i;

        let mut nm = 0.0;
        let mut u = vec![0.0;n1];

        for i1 in i..n {
            nm += r[i1*m+i]*r[i1*m+i];
            u[i1-i] = r[i1*m+i];
        }

        u[0] -= sgn(r[i*m+i])*my_sqrt(nm);
        let z = my_sqrt(nm-r[i*m+i]*r[i*m+i]+u[0]*u[0]);

        if z > 0.0 {
            for i1 in 0..n1 {
                u[i1] = u[i1]/z;
            }

            let mut r1 = vec![0.0;m-i];
            for i1 in i..n {
                for j1 in i..m {
                    r1[j1-i] += u[i1-i]*r[i1*m+j1];
                }
            }

            for i1 in i..n {
                for j1 in i..m {
                    r[i1*m+j1] -= 2.0*u[i1-i]*r1[j1-i];
                }
            }

            let mut q1 = vec![0.0;n];
            for i1 in i..n {
                for j1 in 0..n {
                    q1[j1] += u[i1-i]*q_lt[i1*n+j1];
                }
            }

            for i1 in i..n {
                for j1 in 0..n {
                    q_lt[i1*n+j1] -= 2.0*u[i1-i]*q1[j1];
                }
            }
        }

        if m-i-1 > 0 {
            let n1 = m-i-1;

            let mut nm = 0.0;
            let mut u = vec![0.0;n1];

            for j1 in i+1..m {
                nm += r[i*m+j1]*r[i*m+j1];
                u[j1-i-1] = r[i*m+j1];
            }

            u[0] -= sgn(r[i*m+i+1])*my_sqrt(nm);
            let z = my_sqrt(nm-r[i*m+i+1]*r[i*m+i+1]+u[0]*u[0]);

            if z > 0.0 {
                for i1 in 0..n1 {
                    u[i1] = u[i1]/z;
                }
                
                let mut r1 = vec![0.0;n-i];
                for i1 in i..n {
                    for j1 in i+1..m {
                        r1[i1-i] += u[j1-i-1]*r[i1*m+j1];
                    }
                }

                for i1 in i..n {
                    for j1 in i+1..m {
                        r[i1*m+j1] -= 2.0*u[j1-i-1]*r1[i1-i];
                    }
                }
                
                let mut q1 = vec![0.0;m];
                for i1 in 0..m {
                    for j1 in i+1..m {
                        q1[i1] += u[j1-i-1]*q_rt[i1*m+j1];
                    }
                }

                for i1 in 0..m {
                    for j1 in i+1..m {
                        q_rt[i1*m+j1] -= 2.0*u[j1-i-1]*q1[i1];
                    }
                }
            }
        }
    }

    return (q_lt, r, q_rt);
}

pub fn eigenvalue_bidiagonal(a:&[f64], n:usize, m:usize, i1:usize, i2:usize, j1:usize, j2:usize) -> f64{
    let h = min(i2, j2)+1;

    let mut d1 = 0.0;
    let mut d2 = 0.0;
    let mut d3 = 0.0;
    let mut d4 = 0.0;

    if h >= 3 {
        d1 = a[(h-3)*m+h-2];
    }

    if h >= 2 {
        d2 = a[(h-2)*m+h-2];
        d3 = a[(h-2)*m+h-1];
    }

    if h >= 1 {
        d4 = a[(h-1)*m+h-1];
    }

    
    let a1 = d2*d2 + d1*d1;
    let a2 = d2*d3;
    let a3 = d2*d3;
    let a4 = d4*d4 + d3*d3;

    let u = 1.0;
    let b = -(a1+a4);
    let c = a1*a4-a2*a3;

    let v1 = (-b + my_sqrt(b*b-4.0*u*c))/(2.0*u);
    let v2 = (-b - my_sqrt(b*b-4.0*u*c))/(2.0*u);

    if (v1-a4).abs() < (v2-a4).abs() {
        return v1;
    }
        
    return v2;
}

pub fn givens_right_rotation(a:&[f64], _n:usize, m:usize, i:usize, j:usize, flip:bool) -> (f64, f64) {
    let x = a[i*m+j-1];
    let y = a[i*m+j];
    let w = x*x+y*y;
    let r;

    if w < 1e-100 {
        r = hypot(x, y);
    }
    else {
        r = my_sqrt(w);
    }

    if flip {
        return (y/r, -x/r);
    }

    return (x/r, -y/r);
}

pub fn givens_right_rotation_multiply(a:&mut [f64], n:usize, m:usize, c:f64, s:f64, _i:usize, j:usize, r1:usize, r2:usize, c1:usize, c2:usize) {
    for i1 in r1..r2+1 {
        let p = a[i1*m+j-1];
        let q = a[i1*m+j];
        a[i1*m+j-1] = c*p - s*q;
        a[i1*m+j] = s*p + c*q;
    }
}

pub fn givens_left_rotation(a:&[f64], _n:usize, m:usize, i:usize, j:usize, flip:bool) -> (f64, f64) {
    let x = a[(i-1)*m+j];
    let y = a[i*m+j];
    let w = x*x+y*y;
    let r;
    
    if w < 1e-100 {
        r = hypot(x, y);
    }
    else {
        r = my_sqrt(w);
    }

    if flip {
        return (y/r, -x/r);
    }

    return (x/r, -y/r);
}

pub fn givens_left_rotation_multiply(a:&mut [f64], _n:usize, m:usize, c:f64, s:f64, i:usize, _j:usize, r1:usize, r2:usize, c1:usize, c2:usize) {
    for j1 in c1..c2+1 {
        let p = a[(i-1)*m+j1];
        let q = a[i*m+j1];
        a[(i-1)*m+j1] = c*p - s*q;
        a[i*m+j1] = s*p + c*q;
    }
}

pub fn givens_rotation_qr(a:&[f64], n:usize, m:usize,) -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    if n < m {
        let mut r = transpose(&a, n, m);
        let mut q = identity(m);

        for j in 0..n {
            for i in (j+1..m).rev() {
                let b = givens_left_rotation(&r, m, n, i, j, false);
                givens_left_rotation_multiply(&mut r, m, n, b.0, b.1, i, j, 0, m-1, 0, n-1);
                givens_left_rotation_multiply(&mut q, m, m, b.0, b.1, i, j, 0, m-1, 0, m-1);
            }
        }

        return (transpose(&r, m, n), q, n, m, m);
    }
    else {
        let mut r = a.to_vec();
        let mut q = identity(n);

        for j in 0..m {
            for i in (j+1..n).rev() {
                let b = givens_left_rotation(&r, n, m, i, j, false);
                givens_left_rotation_multiply(&mut r, n, m, b.0, b.1, i, j, 0, n-1, 0, m-1);
                givens_left_rotation_multiply(&mut q, n, n, b.0, b.1, i, j, 0, n-1, 0, n-1);
            }
        }

        return (transpose(&q, n, n), r, n, n, m);
    }
    
}

pub fn householder_reflection_qr(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    if n < m {
        let a1 = transpose(&a, n, m);
        let (q, r) = householder_reflection_left_multiply(&a1, m, n);
        return (transpose(&r, m, n), q, n, m, m);
    }
    else {
        let (q, r) = householder_reflection_left_multiply(&a, n, m);
        return (transpose(&q, n, n), r, n, n, m);
    }
}

pub fn qr(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let qrd = givens_rotation_qr(&a, n, m);
    let mut q = qrd.0;
    let r = qrd.1;

    transpose(&mut q, n, n);
    
    if m < n {
        let x = sub_mat(&q, n, n, 0, n-1, 0, m-1);
        let y = sub_mat(&r, n, m, 0, m-1, 0, m-1);
        return (x, y);
    }

    return (q, r);
}

pub fn golub_kahan(a:&mut [f64], l:&mut [f64], r:&mut [f64], n:usize, m:usize, z:usize, i:usize, j:usize) {
    let mu = eigenvalue_bidiagonal(&a, z, z, i, j, i, j);
    
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

    // for i in 0..n {
    //     a[i*m+2] = a[i*m+3];
    // }

    // let a_s = SparseMatrix::create(n, m, &a);
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



