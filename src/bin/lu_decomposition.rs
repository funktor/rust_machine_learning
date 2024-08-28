#![feature(portable_simd)]
use std::{f64::MIN, simd::prelude::*};
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

fn matrix_multiply_simd(inp1:&Vec<f64>, inp2:&Vec<f64>, n:usize, m:usize, p:usize) -> Vec<f64> {
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

fn swap_rows(inp:&mut Vec<f64>, m:usize, p:usize, q:usize) {
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

fn reduce_row(inp:&mut Vec<f64>, m:usize, p:usize, q:usize, h:f64) {
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

fn lu_decomposition(eye:&mut Vec<f64>, l:&mut Vec<f64>, u:&mut Vec<f64>, n:usize, m:usize) {
    for j in 0..m {
        let mut mmax:f64 = MIN;
        let mut mmax_i:usize = j;

        for i in j..n {
            if u[i*m+j] > mmax {
                mmax = u[i*m+j];
                mmax_i = i;
            }
        }

        if mmax_i != j {
            swap_rows(u, m, j, mmax_i);
            swap_rows(eye, n, j, mmax_i);
            swap_rows(l, n, j, mmax_i);
        }

        for i in j+1..n {
            if u[i*m + j] != 0.0 {
                let h = u[i*m+j]/u[j*m+j];
                reduce_row(u, m, j, i, h);
                l[i*n+j] = h;
            }
        }
    }

    for i in 0..n {
        l[i*n+i] = 1.0;
    }
}
fn main() {
    let n = 1234;
    let m = 569;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];
    let mut u:Vec<f64> = vec![0.0;n*m];
    let mut l:Vec<f64> = vec![0.0;n*n];
    let mut eye:Vec<f64> = vec![0.0;n*n];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
        u[i] = a[i];
    }

    for i in 0..n {
        eye[i*n+i] = 1.0;
    }

    lu_decomposition(&mut eye, &mut l, &mut u, n, m);

    let x = matrix_multiply_simd(&eye, &a, n, n, m);
    let y = matrix_multiply_simd(&l, &u, n, n, m);

    for i in 0..n*m {
        if (x[i]-y[i]).abs()/x[i] > 0.01  {
            println!("{:?}, {:?}, {:?}", i, x[i], y[i]);
        }
    }

}
