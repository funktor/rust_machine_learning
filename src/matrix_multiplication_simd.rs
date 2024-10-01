// Matrix multiplication using SIMD in Rust
// Requires Rust nightly build and AVX 512 enabled for best performance.

// > rustup override set nightly
// > export RUSTFLAGS="-C target-cpu=native"
// > cargo install cargo-simd-detect --force
// > cargo simd-detect
// > RUSTFLAGS=-g cargo build --release
// > cargo run --bin matrix_multiplication_simd
#![allow(dead_code)]
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::time::SystemTime;

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

pub fn matrix_multiply_simd_slices(inp1:&[f64], inp2:&[f64], _n:usize, m:usize, p:usize, x11:usize, x12:usize, y11:usize, y12:usize, x21:usize, x22:usize, y21:usize, y22:usize) -> Vec<f64> {
    const LANES:usize = 64;

    let n1 = x12-x11+1;
    let m1 = y12-y11+1;

    let n2 = x22-x21+1;
    let m2 = y22-y21+1;

    let mut out:Vec<f64> = vec![0.0;n1*m2];

    if m1 == n2 {
        for i in x11..x12+1 {
            let mut l = x21;
            for k in y11..y12+1 {
                let a:Simd<f64, LANES> = Simd::splat(inp1[i*m+k]);
                for j in (y21..y22+1).step_by(LANES) {
                    if l*p+j+LANES > (l+1)*p {
                        let mut r:usize = (i-x11)*p+(j-y21);
                        for h in l*p+j..(l+1)*p {
                            out[r] += inp1[i*m+k]*inp2[h];
                            r += 1;
                        }
                    }
                    else {
                        let x:Simd<f64, LANES> = Simd::from_slice(&inp2[l*p+j..l*p+j+LANES]);
                        let z:Simd<f64, LANES> = a*x;
                        let mut c:Simd<f64, LANES> = Simd::from_slice(&out[(i-x11)*p+(j-y21)..(i-x11)*p+(j-y21)+LANES]);
                        c += z;
                        Simd::copy_to_slice(c, &mut out[(i-x11)*p+(j-y21)..(i-x11)*p+(j-y21)+LANES]);
                    }
                }
                l += 1;
            }
        }
    }

    return out;
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
