#![allow(dead_code)]
use std::cmp::min;
use crate::matrix_utils::*;

pub fn householder_reflection_left_multiply(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>) {
    let mut q_lt = identity(n);
    let mut r = a.to_vec();
    let w = min(n, m);

    for i in 0..w {
        let n1 = n-i;

        let mut nm = 0.0;
        let mut u = vec![0.0;n1];
        let r1;
        let q1;

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

            r1 = matrix_multiply_simd_on_slices_left_vector(&u, &r, m,  0, n1-1, i, n-1, i, m-1);

            for i1 in i..n {
                mul_sub_const(&r1, &mut r[i1*m+i..(i1+1)*m], 2.0*u[i1-i], m-i);
            }

            q1 = matrix_multiply_simd_on_slices_left_vector(&u, &q_lt, n,  0, n1-1, i, n-1, 0, n-1);

            for i1 in i..n {
                mul_sub_const(&q1, &mut q_lt[i1*n..(i1+1)*n], 2.0*u[i1-i], n);
            }
        }
    }

    return (q_lt, r);
}

pub fn householder_reflection_bidiagonalization(a:&[f64], n:usize, m:usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut q_lt = identity(n);
    let mut q_rt = identity(m);

    let mut r = a.to_vec();

    for i in 0..min(n, m) {
        let n1 = n-i;

        let mut nm = 0.0;
        let mut u = vec![0.0;n1];
        let mut r1;
        let mut q1;

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

            r1 = matrix_multiply_simd_on_slices_left_vector(&u, &r, m,  0, n1-1, i, n-1, i, m-1);

            for i1 in i..n {
                mul_sub_const(&r1, &mut r[i1*m+i..(i1+1)*m], 2.0*u[i1-i], m-i);
            }

            q1 = matrix_multiply_simd_on_slices_left_vector(&u, &q_lt, n,  0, n1-1, i, n-1, 0, n-1);

            for i1 in i..n {
                mul_sub_const(&q1, &mut q_lt[i1*n..(i1+1)*n], 2.0*u[i1-i], n);
            }
        }

        if m-i-1 > 0 {
            let n1 = m-i-1;

            nm = 0.0;
            u = vec![0.0;n1];

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

                r1 = matrix_multiply_simd_on_slices_right_vector(&r, &u, m, i, n-1, i+1, m-1);

                for i1 in i..n {
                    for j1 in i+1..m {
                        r[i1*m+j1] -= 2.0*u[j1-i-1]*r1[i1-i];
                    }
                }

                q1 = matrix_multiply_simd_on_slices_right_vector(&q_rt, &u, m, 0, m-1, i+1, m-1);

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

pub fn givens_right_rotation_multiply(a:&mut [f64], _n:usize, m:usize, c:f64, s:f64, _i:usize, j:usize, r1:usize, r2:usize, _c1:usize, _c2:usize) {
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

pub fn givens_left_rotation_multiply(a:&mut [f64], _n:usize, m:usize, c:f64, s:f64, i:usize, _j:usize, _r1:usize, _r2:usize, c1:usize, c2:usize) {
    for j1 in c1..c2+1 {
        let p = a[(i-1)*m+j1];
        let q = a[i*m+j1];
        a[(i-1)*m+j1] = c*p - s*q;
        a[i*m+j1] = s*p + c*q;
    }
}