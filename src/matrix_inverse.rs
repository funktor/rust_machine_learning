#![allow(dead_code)]
use crate::solve_linear::solve;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn inverse(inp:&[f64], n:usize) -> Vec<f64>{
    let mut sol:Vec<f64> = vec![0.0;n*n];

    for i in 0..n {
        let mut b:Vec<f64> = vec![0.0;n];
        b[i] = 1.0;
        let x = solve(inp, &b, n);
        for j in 0..n {
            sol[j*n+i] = x[j];
        }
    }

    return sol;
}

pub fn run() {
    let n = 3;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*n];

    for i in 0..n*n {
        a[i] = normal.sample(&mut rng);
    }

    let sol = inverse(&a, n);

    println!("{:?}", a);
    println!("{:?}", sol);
}