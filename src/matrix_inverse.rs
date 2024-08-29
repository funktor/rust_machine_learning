#![allow(dead_code)]
use crate::solve_linear::solve;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use std::thread;
use std::sync::Arc;
use std::cmp::min;

pub fn inverse(inp:&[f64], n:usize) -> Vec<f64>{
    let mut sol:Vec<f64> = vec![0.0;n*n];
    let a = Arc::new(inp.to_owned());

    let mut handles = vec![];
    let q = (n as f64/4.0).ceil() as usize;

    for i in (0..n).step_by(q) {
        let y = Arc::clone(&a);
        let mut results = vec![];
        
        let handle = thread::spawn(move || {
            for j in i..min(i+q, n) {
                let mut b:Vec<f64> = vec![0.0;n];
                b[j] = 1.0;
                let x = solve(&y, &b, n);
                results.push((x, j));
            }

            return results;
        });

        handles.push(handle);
    }

    for handle in handles {
        let res = handle.join().unwrap();
        for out in res {
            for j in 0..n {
                sol[j*n+out.1] = out.0[j];
            }
        }
    }

    return sol;
}

pub fn run() {
    let n = 500;

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