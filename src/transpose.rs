#![allow(dead_code)]
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub fn transpose(a:&[f64], n:usize, m:usize) -> Vec<f64> {
    let mut b = a.to_vec();
    
    for i in 0..n {
        for j in i+1..m {
            let k = b[i*m+j];
            b[i*m+j] = b[j*n+i];
            b[j*n+i] = k;
        }
    }

    return b;
}

pub fn run() {
    let n = 145;
    let m = 145;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
    }

    transpose(&mut a, n, m);
    println!("{:?}", a);
}
