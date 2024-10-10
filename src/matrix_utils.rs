#![allow(dead_code)]
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::time::SystemTime;
use std::cmp::min;

pub fn copy(a:&[f64], b:&mut [f64], n:usize) {
    const LANES:usize = 64;

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                b[j] = a[j];
            }
        }
        else {
            let x:Simd<f64, LANES> = Simd::from_slice(&a[i..i+LANES]);
            Simd::copy_to_slice(x, &mut b[i..i+LANES]);
        }
    }
}

pub fn dot_product(inp1:&[f64], inp2:&[f64]) -> f64 {
    let n:usize = inp1.len();
    let mut sum:f64 = 0.0;
    for i in 0..n {
        sum += inp1[i]*inp2[i];
    }

    return sum;
}

pub fn dot_product_simd(inp1:&[f64], inp2:&[f64]) -> f64 {
    const LANES:usize = 64;
    let n:usize = inp1.len();
    let mut sum:f64 = 0.0;

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            sum += dot_product(&inp1[i..n].to_vec(), &inp2[i..n].to_vec());
            break;
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp1[i..i+LANES]);
            let b:Simd<f64, LANES> = Simd::from_slice(&inp2[i..i+LANES]);
            let c = a*b;
            sum += c.reduce_sum();
        }
    }

    return sum;
}

pub fn matrix_multiply(inp1:&[f64], inp2:&[f64], n:usize, m:usize, p:usize) -> Vec<f64> {
    let mut out:Vec<f64> = vec![0.0;n*p];
    for i in 0..n {
        for k in 0..m {
            for j in 0..p {
                out[i*p+j] += inp1[i*m+k]*inp2[k*p+j];
            }
        }
    }

    return out;
}

pub fn matrix_multiply_simd(inp1:&[f64], inp2:&[f64], n:usize, m:usize, p:usize) -> Vec<f64> {
    const LANES:usize = 64;
    let mut out:Vec<f64> = vec![0.0;n*p];

    for i in 0..n {
        for k in 0..m {
            let a:Simd<f64, LANES> = Simd::splat(inp1[i*m+k]);
            for j in (0..p).step_by(LANES) {
                if k*p+j+LANES > (k+1)*p {
                    let mut r:usize = i*p+j;
                    for h in k*p+j..(k+1)*p {
                        out[r] += inp1[i*m+k]*inp2[h];
                        r += 1;
                    }
                }
                else {
                    let x:Simd<f64, LANES> = Simd::from_slice(&inp2[k*p+j..k*p+j+LANES]);
                    let z:Simd<f64, LANES> = a*x;
                    let mut c:Simd<f64, LANES> = Simd::from_slice(&out[i*p+j..i*p+j+LANES]);
                    c += z;
                    Simd::copy_to_slice(c, &mut out[i*p+j..i*p+j+LANES]);
                }
            }
        }
    }
    return out;
}

pub fn matrix_multiply_simd_on_slices(inp1:&[f64], inp2:&[f64], _n1:usize, m1:usize, _n2:usize, m2:usize, x11:usize, x12:usize, y11:usize, y12:usize, x21:usize, x22:usize, y21:usize, y22:usize) -> Vec<f64> {
    const LANES:usize = 64;

    let q1 = x12-x11+1;
    let r1 = y12-y11+1;

    let q2 = x22-x21+1;
    let r2 = y22-y21+1;

    let mut out:Vec<f64> = vec![0.0;q1*r2];

    if r1 == q2 {
        for i in x11..x12+1 {
            let mut l = x21;
            for k in y11..y12+1 {
                let a:Simd<f64, LANES> = Simd::splat(inp1[i*m1+k]);
                for j in (y21..y22+1).step_by(LANES) {
                    if j+LANES > y22+1 {
                        let mut r:usize = (i-x11)*r2+(j-y21);
                        for h in l*m2+j..l*m2+y22+1 {
                            out[r] += inp1[i*m1+k]*inp2[h];
                            r += 1;
                        }
                    }
                    else {
                        let x:Simd<f64, LANES> = Simd::from_slice(&inp2[l*m2+j..l*m2+j+LANES]);
                        let z:Simd<f64, LANES> = a*x;
                        let mut c:Simd<f64, LANES> = Simd::from_slice(&out[(i-x11)*r2+(j-y21)..(i-x11)*r2+(j-y21)+LANES]);
                        c += z;
                        Simd::copy_to_slice(c, &mut out[(i-x11)*r2+(j-y21)..(i-x11)*r2+(j-y21)+LANES]);
                    }
                }
                l += 1;
            }
        }
    }

    return out;
}

pub fn matrix_multiply_simd_on_slices_left_vector(inp1:&[f64], inp2:&[f64], m:usize, y11:usize, y12:usize, x21:usize, x22:usize, y21:usize, y22:usize) -> Vec<f64> {
    const LANES:usize = 64;

    let r1 = y12-y11+1;

    let q2 = x22-x21+1;
    let r2 = y22-y21+1;

    let mut out:Vec<f64> = vec![0.0;r2];

    if r1 == q2 {
        let mut l = x21;
        for k in y11..y12+1 {
            let a:Simd<f64, LANES> = Simd::splat(inp1[k]);
            for j in (y21..y22+1).step_by(LANES) {
                if j+LANES > y22+1 {
                    let mut r:usize = j-y21;
                    for h in l*m+j..l*m+y22+1 {
                        out[r] += inp1[k]*inp2[h];
                        r += 1;
                    }
                }
                else {
                    let x:Simd<f64, LANES> = Simd::from_slice(&inp2[l*m+j..l*m+j+LANES]);
                    let z:Simd<f64, LANES> = a*x;
                    let mut c:Simd<f64, LANES> = Simd::from_slice(&out[j-y21..j-y21+LANES]);
                    c += z;
                    Simd::copy_to_slice(c, &mut out[j-y21..j-y21+LANES]);
                }
            }
            l += 1;
        }
    }

    return out;
}

pub fn matrix_multiply_simd_on_slices_right_vector(inp1:&[f64], inp2:&[f64], m:usize, x11:usize, x12:usize, y11:usize, y12:usize) -> Vec<f64> {
    let mut out = vec![0.0;x12-x11+1];

    for i in x11..x12+1 {
        out[i-x11] = dot_product_simd(&inp1[i*m+y11..i*m+y12+1], &inp2);
    }

    return out;
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

pub fn sgn(x:f64) -> f64 {
    if x < 0.0 {
        return -1.0;
    }
    return 1.0;
}

pub fn identity(n:usize) -> Vec<f64> {
    let mut q = vec![0.0;n*n];
    for i in 0..n {
        q[i*(n+1)] = 1.0;
    }

    return q;
}

pub fn my_sqrt(x:f64) -> f64 {
    if x > 0.0 {
        return x.sqrt().abs();
    }
    return 0.0;
}

pub fn hypot(x:f64, y:f64) -> f64 {
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

pub fn mul_sub_const(inp:&[f64], out:&mut [f64], x:f64, n:usize) {
    const LANES:usize = 64;
    let b = Simd::splat(x);

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                out[j] -= inp[j]*x;
            }
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp[i..i+LANES]);
            let y = a*b;
            let mut c:Simd<f64, LANES> = Simd::from_slice(&out[i..i+LANES]);
            c -= y;
            Simd::copy_to_slice(c, &mut out[i..i+LANES]);
        }
    }
}

pub fn eigenvalue_bidiagonal_slices(a:&[f64], _n:usize, m:usize, _i1:usize, i2:usize, _j1:usize, j2:usize) -> f64{
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

pub fn transpose(a:&[f64], n:usize, m:usize) -> Vec<f64> {
    let mut b = vec![0.0;n*m];

    for i in 0..n {
        for j in 0..m {
            b[j*n+i] = a[i*m+j];
        }
    }

    return b;
}

pub fn run() {
    // Matrix multiplication
    let n:usize = 101;
    let m:usize = 511;
    let p:usize = 397;
    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    let mut inp1:Vec<f64> = vec![0.0;n*m];
    let mut inp2:Vec<f64> = vec![0.0;m*p];

    for i in 0..n*m {
        inp1[i] = normal.sample(&mut rng);
    }

    for i in 0..m*p {
        inp2[i] = normal.sample(&mut rng);
    }

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();
    let prod1 = matrix_multiply_simd(&inp1, &inp2, n, m, p);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();

    println!("{:?}", end_time-start_time);

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();
    let prod2 = matrix_multiply(&inp1, &inp2, n, m, p);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();

    println!("{:?}", end_time-start_time);

    assert!(prod1 == prod2, "Matrix multiplications results are different");

}
