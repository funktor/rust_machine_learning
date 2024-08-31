#![allow(dead_code)]
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn swap_rows(inp:&mut [f64], m:usize, p:usize, q:usize) {
    const LANES:usize = 64;

    for j in (0..m).step_by(LANES) {
        if p*m+j+LANES > (p+1)*m {
            for k in j..m {
                let temp = inp[p*m+k];
                inp[p*m+k] = inp[q*m+k];
                inp[q*m+k] = temp;
            }
            break;
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp[p*m+j..p*m+j+LANES]);
            let b:Simd<f64, LANES> = Simd::from_slice(&inp[q*m+j..q*m+j+LANES]);
            
            Simd::copy_to_slice(a, &mut inp[q*m+j..q*m+j+LANES]);
            Simd::copy_to_slice(b, &mut inp[p*m+j..p*m+j+LANES]);
        }
    }
}

pub fn normalize_row(inp:&mut [f64], h:f64, m:usize, p:usize) {
    const LANES:usize = 64;
    let x:Simd<f64, LANES> = Simd::splat(1.0/h);

    for j in (0..m).step_by(LANES) {
        if p*m+j+LANES > (p+1)*m {
            for k in j..m {
                inp[p*m+k] = inp[p*m+k]/h;
            }
            break;
        }
        else {
            let mut a:Simd<f64, LANES> = Simd::from_slice(&inp[p*m+j..p*m+j+LANES]);
            a *= x;          
            Simd::copy_to_slice(a, &mut inp[p*m+j..p*m+j+LANES]);
        }
    }
}

pub fn reduce_row(inp:&mut [f64], m:usize, p:usize, q:usize, h:f64) {
    const LANES:usize = 64;
    let x:Simd<f64, LANES> = Simd::splat(h);

    for j in (0..m).step_by(LANES) {
        if p*m+j+LANES > (p+1)*m {
            for k in j..m {
                inp[q*m+k] = inp[q*m+k] - h*inp[p*m+k];
            }
            break;
        }
        else {
            let a:Simd<f64, LANES> = Simd::from_slice(&inp[p*m+j..p*m+j+LANES]);
            let b:Simd<f64, LANES> = Simd::from_slice(&inp[q*m+j..q*m+j+LANES]);
            let c = b-a*x;
            
            Simd::copy_to_slice(c, &mut inp[q*m+j..q*m+j+LANES]);
        }
    }
}

pub fn row_echelon(inp:&mut [f64], n:usize, m:usize) {
    for j in 0..m {
        for i in j..n {
            if inp[i*m + j] != 0.0 {
                let h = inp[i*m+j];
                if i == j {
                    normalize_row(inp, h, m, j);
                }
                else if i > j && inp[j*m+j] != 1.0 {
                    swap_rows(inp, m, j, i);
                    normalize_row(inp, h, m, j);
                }
                else {
                    reduce_row(inp, m, j, i, h);
                }
            }
        }
    }
}

pub fn run() {
    let n = 10;
    let m = 5;
    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    let mut mat:Vec<f64> = vec![0.0;n*m];
    for i in 0..n*m {
        mat[i] = normal.sample(&mut rng);
    }

    row_echelon(&mut mat, n, m);
    
    for i in 0..n {
        for j in 0..m {
            print!("{:?} ", mat[i*m+j]);
        }
        println!();
    }
}
