#![allow(dead_code)]
use crate::copy::copy;
use crate::transpose::transpose;
use crate::matrix_multiplication_simd::{matrix_multiply_simd, matrix_multiply_simd_slices};
// use crate::sparse_matrix::*;
use std::simd::prelude::*;
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

fn identity(n:usize) -> Vec<f64> {
    let mut q = vec![0.0;n*n];
    for i in 0..n {
        q[i*(n+1)] = 1.0;
    }

    return q;
}

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

pub fn householder_reflection_bidiagonalization(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut q_lt = identity(n);
    let mut q_rt = identity(m);

    let mut r = a.to_vec();

    for i in 0..min(n, m) {
        let n1 = n-i;

        let mut nm = 0.0;
        for i1 in i..n {
            nm += r[i1*m+i]*r[i1*m+i];
        }

        let mut v = vec![0.0;n1];
        let mut z = 0.0;
        for i1 in i..n {
            if i1 == i {
                v[i1-i] = r[i1*m+i]-sgn(r[i*m+i])*nm.sqrt();
            }
            else {
                v[i1-i] = r[i1*m+i];
            }
            z += v[i1-i]*v[i1-i];
        }

        z = z.sqrt();

        if z > 0.0 {
            let u = mul_const(&v, 1.0/z, n1);

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

            let mut q1 = vec![0.0;m];
            for i1 in i..n {
                for j1 in 0..m {
                    q1[j1] += u[i1-i]*q_lt[i1*m+j1];
                }
            }

            for i1 in i..n {
                for j1 in 0..m {
                    q_lt[i1*m+j1] -= 2.0*u[i1-i]*q1[j1];
                }
            }
        }

        if m-i-1 > 0 {
            let n1 = m-i-1;

            let mut nm = 0.0;
            for j1 in i+1..m {
                nm += r[i*m+j1]*r[i*m+j1];
            }

            let mut v = vec![0.0;n1];
            let mut z = 0.0;
            for j1 in i+1..n {
                if j1 == i+1 {
                    v[j1-i-1] = r[i*m+j1]-sgn(r[i*m+j1])*nm.sqrt();
                }
                else {
                    v[j1-i-1] = r[i*m+j1];
                }
                z += v[j1-i-1]*v[j1-i-1];
            }

            z = z.sqrt();

            if z > 0.0 {
                let u = mul_const(&v, 1.0/z, n1);

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
                
                let mut q1 = vec![0.0;n];
                for i1 in 0..n {
                    for j1 in i+1..m {
                        q1[i1] += u[j1-i-1]*q_rt[i1*m+j1];
                    }
                }

                for i1 in 0..n {
                    for j1 in i+1..m {
                        q_rt[i1*m+j1] -= 2.0*u[j1-i-1]*q1[i1];
                    }
                }
            }
        }
    }

    return (q_lt, r, q_rt);
}

pub fn eigenvalue_bidiagonal(a:&[f64], n:usize, m:usize) -> f64{
    let h = min(n, m);

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

    let v1 = (-b + (b*b-4.0*u*c).sqrt())/(2.0*u);
    let v2 = (-b - (b*b-4.0*u*c).sqrt())/(2.0*u);

    if (v1-a4).abs() < (v2-a4).abs() {
        return v1;
    }
        
    return v2;
}

pub fn givens_right_rotation(a:&[f64], _n:usize, m:usize, i:usize, j:usize) -> (f64, f64) {
    let x = a[i*m+j-1];
    let y = a[i*m+j];
    let r = (x*x + y*y).sqrt();

    return (x/r, -y/r);
}

pub fn givens_right_rotation_multiply(a:&mut [f64], n:usize, m:usize, c:f64, s:f64, _i:usize, j:usize) {
    let b = vec![c, s, -s, c];
    let d = matrix_multiply_simd_slices(&a, &b, n, m, 2, 0, n-1, j-1, j, 0, 1, 0, 1);
    copy_sub_mat(a, &d, n, m, 0, n-1, j-1, j);
}

pub fn givens_left_rotation(a:&[f64], _n:usize, m:usize, i:usize, j:usize) -> (f64, f64) {
    let x = a[(i-1)*m+j];
    let y = a[i*m+j];
    let r = (x*x + y*y).sqrt();

    return (x/r, -y/r);
}

pub fn givens_left_rotation_multiply(a:&mut [f64], _n:usize, m:usize, c:f64, s:f64, i:usize, _j:usize) {
    let b = vec![c, -s, s, c];
    let d = matrix_multiply_simd(&b, &a[(i-1)*m..(i+1)*m], 2, 2, m);
    copy(&d, &mut a[(i-1)*m..(i+1)*m], 2*m);
}

pub fn givens_rotation_qr(a:&[f64], n:usize, m:usize,) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0;n*n];

    for i in 0..n {
        q[i*(n+1)] = 1.0;
    }

    let mut r = a.to_vec();

    for j in 0..m {
        for i in (j+1..n).rev() {
            if r[i*m+j].abs() > 1e-10 {
                let b = givens_left_rotation(&r, n, m, i, j);
                givens_left_rotation_multiply(&mut r, n, m, b.0, b.1, i, j);
                givens_left_rotation_multiply(&mut q, n, n, b.0, b.1, i, j);
            }
        }
    }

    return (q, r);
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
    let mu = eigenvalue_bidiagonal(&sub_mat(&a, z, z, i, j, i, j), j-i+1, j-i+1);
    
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

        let b = givens_right_rotation(&a, z, z, x, y);

        if k == i {
            a[i*z+i] = u;
            a[i*z+i+1] = v;
        }

        givens_right_rotation_multiply(a, z, z, b.0, b.1, x, y);
        givens_right_rotation_multiply(r, m, z, b.0, b.1, x, y);

        if k > i {
            x = k+1;
            y = k;
        }
        else {
            x = i+1;
            y = i;
        }
            
        let b = givens_left_rotation(&a, z, z, x, y);

        givens_left_rotation_multiply(a, z, z, b.0, b.1, x, y);
        givens_left_rotation_multiply(l, z, n, b.0, b.1, x, y);
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
    
    let eps = 1.0e-10;
    
    loop {
        for i in 0..r-1 {
            if a1[i*r+i+1].abs() < eps*(a1[i*r+i].abs() + a1[(i+1)*r+i+1].abs()) {
                a1[i*r+i+1] = 0.0;
            }
        }

        let mut q = 0;
        for i in (0..r-1).rev() {
            if a1[i*r+i+1].abs() > eps {
                q = i+1;
                break;
            }
        }

        if q == 0 {
            break;
        }

        let mut p = 0;
        for i in (0..q).rev() {
            if a1[i*r+i+1].abs() <= eps {
                p = i+1;
                break;
            }
        }

        let mut flag: bool = false;
        
        for i in p..q {
            if a1[i*r+i].abs() <= eps {
                flag = true;
                for j in i+1..r {
                    let b = givens_left_rotation(&a1, r, r, i, j);
                    
                    givens_left_rotation_multiply(&mut a1, r, r, b.0, b.1, i, j);
                    givens_left_rotation_multiply(&mut u, r, n, b.0, b.1, i, j);
                }

                for j in (0..i).rev() {
                    let b = givens_right_rotation(&a1, r, r, i, j);

                    givens_right_rotation_multiply(&mut a1, r, r, b.0, b.1, i, j);
                    givens_right_rotation_multiply(&mut v, m, r, b.0, b.1, i, j);
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
    let (q, r) = qr(&b, n, l);
    let c = &matrix_multiply_simd(&transpose(&q, n, n), &a, n, n, m);
    let (mut u, s, v) = golub_reisch_svd(&c, n, m);
    u = matrix_multiply_simd(&q, &u, n, n, n);
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

    // let a_s = SparseMatrix::create(n, m, &a);
    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    let b = householder_reflection_bidiagonalization(&a, n, m);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    println!("{:?}", end_time-start_time);
    // let r = min(n, m);
    // let mut c = matrix_multiply_simd(&transpose(&b.0, n, n), &b.1, n, n, m);
    // c = matrix_multiply_simd(&c, &transpose(&b.2, m, m), n, m, m);

    // println!("{:?}", a);
    // println!();
    // println!("{:?}", b.1);
    // println!();
    // println!("{:?}", c);
}



