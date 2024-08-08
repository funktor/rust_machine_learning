// Dot product implementation using SIMD in Rust
// Requires Rust nightly build and AVX 512 enabled for best performance.

// > rustup override set nightly
// > export RUSTFLAGS="-C target-cpu=native"
// > cargo install cargo-simd-detect --force
// > cargo simd-detect
// > RUSTFLAGS=-g cargo build --release
// > cargo run --bin dot_product_simd

#![feature(portable_simd)]
use std::simd::prelude::*;
use rand_distr::{num_traits::abs, Distribution, Normal};
use rand::thread_rng;
use std::time::SystemTime;

fn dot_product(inp1:&Vec<f64>, inp2:&Vec<f64>) -> f64 {
    let n:usize = inp1.len();
    let mut sum:f64 = 0.0;
    for i in 0..n {
        sum += inp1[i]*inp2[i];
    }

    return sum;
}

fn dot_product_simd(inp1:&Vec<f64>, inp2:&Vec<f64>) -> f64 {
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

fn main() {
    // Dot product
    const N:usize = 1234567;
    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(100.0, 5.0).ok().unwrap();
    let mut inp1:Vec<f64> = vec![0.0;N];
    let mut inp2:Vec<f64> = vec![0.0;N];

    for i in 0..N {
        inp1[i] = normal.sample(&mut rng);
        inp2[i] = normal.sample(&mut rng);
    }

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();
    let dp1 = dot_product_simd(&inp1, &inp2);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();

    println!("{:?}", dp1);
    println!("{:?}", end_time-start_time);

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();
    let dp2 = dot_product(&inp1, &inp2);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_micros();

    println!("{:?}", dp2);
    println!("{:?}", end_time-start_time);
    assert!(abs(dp1-dp2)/abs(dp1) <= 0.001, "Dot product results are more than 0.1% different");

}
