#![allow(dead_code)]
use std::simd::prelude::*;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

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

pub fn run() {
    let n = 145;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n];
    let mut b:Vec<f64> = vec![0.0;n];

    for i in 0..n {
        a[i] = normal.sample(&mut rng);
    }

    copy(&a, &mut b, n);
    assert!(a == b, "Copy not working as expected !!!");
}
