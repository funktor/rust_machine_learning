#![allow(dead_code)]
use std::cmp::min;
use rand_distr::{Distribution, Normal, Uniform};
use rand::thread_rng;
use std::time::SystemTime;
use std::thread;
use std::sync::{Arc, Mutex};
use std::simd::prelude::*;

fn binary_search(arr:&[usize], i:usize) -> usize {
    let n = arr.len();
    if n == 0 {
        return 0;
    }

    let mut lt:usize = 0;
    let mut rt:usize = n-1;

    while lt <= rt {
        let mid = (lt + rt)/2;
        if arr[mid] == i {
            return mid;
        }
        else if arr[mid] > i {
            if mid == 0 {
                return n;
            }
            rt = mid-1;
        }
        else {
            lt = mid+1;
        }
    }

    return n;
}

fn binary_search_next(arr:&[usize], i:usize) -> usize {
    let n = arr.len();
    if n == 0 {
        return 0;
    }

    let mut lt:usize = 0;
    let mut rt:usize = n-1;

    let mut p:usize = n;

    while lt <= rt {
        let mid = (lt + rt)/2;
        if arr[mid] >= i {
            p = mid;
            if mid == 0 {
                return p;
            }
            rt = mid-1;
        }
        else {
            lt = mid+1;
        }
    }

    return p;
}

#[derive(Clone)]
pub struct SparseMatrix {
    pub nrow: usize,
    pub ncol: usize,
    keys: Vec<usize>,
    data: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(
        nrow: usize,
        ncol: usize,
        keys: Vec<usize>,
        data: Vec<f64>,
    ) -> Self {

        Self {
            nrow,
            ncol,
            keys,
            data,
        }
    }
}

impl SparseMatrix {
    pub fn create(
        nrow: usize,
        ncol: usize,
        dense_data: &Vec<f64>
    ) -> Self {

        let mut keys:Vec<usize> = Vec::new();
        let mut data:Vec<f64> = Vec::new();

        for i in 0..nrow {
            for j in 0..ncol {
                let key = i*ncol+j;
                if dense_data[key].abs() > 1e-10 {
                    keys.push(key);
                    data.push(dense_data[key]);
                }
            }
        }

        Self {
            nrow,
            ncol,
            keys,
            data,
        }
    }
}

pub fn loc(a:&SparseMatrix, i:usize, j:usize) -> Option<f64> {
    let k = i*a.ncol+j;
    let p = binary_search(&a.keys, k);
    if p < a.data.len() {
        return Some(a.data[p]);
    }
    return None;
}

pub fn get_sub_mat(a:&SparseMatrix, r_start:usize, r_end:usize, c_start:usize, c_end:usize) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let mut n = 0;
    let mut m = 0;

    if a.nrow >= r_end-r_start+1 && a.ncol >= c_end-c_start+1 {
        let k_start = r_start*a.ncol + c_start;
        let mut h = binary_search_next(&a.keys, k_start);

        n = r_end-r_start+1;
        m = c_end-c_start+1;

        while h < a.keys.len() {
            let key = a.keys[h];
            let d = a.data[h];

            let i = key/a.ncol;
            let j = key % a.ncol;

            if i > r_end {
                break;
            }

            if j >= c_start && j <= c_end {
                keys.push((i-r_start)*m+(j-c_start));
                data.push(d);
                h += 1;
            }

            else if j < c_start{
                let k_start = i*a.ncol + c_start;
                h = binary_search_next(&a.keys, k_start);
            }

            else {
                let k_start = (i+1)*a.ncol + c_start;
                h = binary_search_next(&a.keys, k_start);
            }
        }
    }

    return SparseMatrix::new(n, m, keys, data);
}

pub fn add(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let mut n = 0;
    let mut m = 0;

    if a.nrow > 0 && a.ncol > 0 && a.nrow == b.nrow && a.ncol == b.ncol {
        n = a.nrow;
        m = a.ncol;

        let mut i:usize = 0;
        let mut j:usize = 0;

        while i < a.keys.len() && j < b.keys.len() {
            if a.keys[i] < b.keys[j] {
                keys.push(a.keys[i]);
                data.push(a.data[i]);
                i += 1;
            }
            else if a.keys[i] > b.keys[j] {
                keys.push(b.keys[j]);
                data.push(b.data[j]);
                j += 1;
            }
            else {
                if (a.data[i]+b.data[j]).abs() > 1e-10 {
                    keys.push(a.keys[i]);
                    data.push(a.data[i]+b.data[j]);
                }
                i += 1;
                j += 1;
            }
        }

        while i < a.keys.len() {
            keys.push(a.keys[i]);
            data.push(a.data[i]);
            i += 1;
        }

        while j < b.keys.len() {
            keys.push(b.keys[j]);
            data.push(b.data[j]);
            j += 1;
        }
    }
    
    return SparseMatrix::new(n, m, keys, data);
}

pub fn sub(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let mut n = 0;
    let mut m = 0;

    if a.nrow > 0 && a.ncol > 0 && a.nrow == b.nrow && a.ncol == b.ncol {
        n = a.nrow;
        m = a.ncol;

        let mut i:usize = 0;
        let mut j:usize = 0;

        while i < a.keys.len() && j < b.keys.len() {
            if a.keys[i] < b.keys[j] {
                keys.push(a.keys[i]);
                data.push(a.data[i]);
                i += 1;
            }
            else if a.keys[i] > b.keys[j] {
                keys.push(b.keys[j]);
                data.push(-b.data[j]);
                j += 1;
            }
            else {
                if (a.data[i]-b.data[j]).abs() > 1e-10 {
                    keys.push(a.keys[i]);
                    data.push(a.data[i]-b.data[j]);
                }
                i += 1;
                j += 1;
            }
        }

        while i < a.keys.len() {
            keys.push(a.keys[i]);
            data.push(a.data[i]);
            i += 1;
        }

        while j < b.keys.len() {
            keys.push(b.keys[j]);
            data.push(-b.data[j]);
            j += 1;
        }
    }
    
    return SparseMatrix::new(n, m, keys, data);
}

pub fn mul(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let mut n = 0;
    let mut m = 0;

    if a.nrow > 0 && a.ncol > 0 && a.nrow == b.nrow && a.ncol == b.ncol {
        n = a.nrow;
        m = a.ncol;

        let mut i:usize = 0;
        let mut j:usize = 0;

        while i < a.keys.len() && j < b.keys.len() {
            if a.keys[i] < b.keys[j] {
                i += 1;
            }
            else if a.keys[i] > b.keys[j] {
                j += 1;
            }
            else {
                if (a.data[i]*b.data[j]).abs() > 1e-10 {
                    keys.push(a.keys[i]);
                    data.push(a.data[i]*b.data[j]);
                }
                i += 1;
                j += 1;
            }
        }
    }
    
    return SparseMatrix::new(n, m, keys, data);
}

pub fn mul_const(a:&SparseMatrix, b:f64) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    for i in 0..a.keys.len() {
        if (b*a.data[i]).abs() > 1e-10 {
            keys.push(a.keys[i]);
            data.push(b*a.data[i]);
        }
    }

    return SparseMatrix::new(a.nrow, a.ncol, keys, data);
}

pub fn transpose(a:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();
    let mut zipped:Vec<(usize, f64)> = Vec::new();

    for i in 0..a.keys.len() {
        let key = a.keys[i];
        let d = a.data[i];
        let r = key/a.ncol;
        let c = key % a.ncol;

        zipped.push((c*a.nrow+r, d));
    }

    zipped.sort_by_key(|x| x.0);

    for (key, d) in zipped {
        keys.push(key);
        data.push(d);
    }

    return SparseMatrix::new(a.ncol, a.nrow, keys, data);
}

pub fn copy(a:&SparseMatrix, b:&SparseMatrix, r_start:usize, r_end:usize, c_start:usize, c_end:usize) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let mut n = 0;
    let mut m = 0;

    if a.nrow >= b.nrow && a.ncol >= b.ncol {
        n = a.nrow;
        m = a.ncol;

        let mut b_keys:Vec<usize> = Vec::new();
        let mut b_data:Vec<f64> = Vec::new();

        for i in 0..b.keys.len() {
            let key = b.keys[i];
            let r = key/b.ncol;
            let c = key % b.ncol;
            let nkey = (r+r_start)*a.ncol+(c+c_start);
            b_keys.push(nkey);
            b_data.push(b.data[i]);
        }

        let mut i:usize = 0;
        let mut j:usize = 0;

        while i < a.keys.len() && j < b_keys.len() {
            if a.keys[i] < b_keys[j] {
                let r = a.keys[i]/a.ncol;
                let c = a.keys[i] % a.ncol;

                if r < r_start || r > r_end || c < c_start || c > c_end {
                    keys.push(a.keys[i]);
                    data.push(a.data[i]);
                } 
                
                i += 1;
            }
            else if a.keys[i] > b_keys[j] {
                keys.push(b_keys[j]);
                data.push(b_data[j]);
                j += 1;
            }
            else {
                keys.push(b_keys[j]);
                data.push(b_data[j]);
                i += 1;
                j += 1;
            }
        }

        while i < a.keys.len() {
            let r = a.keys[i]/a.ncol;
            let c = a.keys[i] % a.ncol;

            if r < r_start || r > r_end || c < c_start || c > c_end {
                keys.push(a.keys[i]);
                data.push(a.data[i]);
            }
            i += 1;
        }

        while j < b_keys.len() {
            keys.push(b_keys[j]);
            data.push(b_data[j]);
            j += 1;
        }
    }
    
    return SparseMatrix::new(n, m, keys, data);
}

pub fn vstack(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let n1 = a.keys.len();
    let n2 = b.keys.len();

    let mut keys:Vec<usize> = vec![0;n1+n2];
    let mut data:Vec<f64> = vec![0.0;n1+n2];

    let mut n = 0;
    let mut m = 0;

    if a.ncol == b.ncol {
        n = a.nrow + b.nrow;
        m = a.ncol;

        const LANES:usize = 64;

        for i in (0..n1).step_by(LANES) {
            if i+LANES > n1 {
                for j in i..n1 {
                    keys[j] = a.keys[j];
                    data[j] = a.data[j];
                }
            }
            else {
                let x:Simd<usize, LANES> = Simd::from_slice(&a.keys[i..i+LANES]);
                Simd::copy_to_slice(x, &mut keys[i..i+LANES]);

                let x:Simd<f64, LANES> = Simd::from_slice(&a.data[i..i+LANES]);
                Simd::copy_to_slice(x, &mut data[i..i+LANES]);
            }
        }

        for i in (0..n2).step_by(LANES) {
            if i+LANES > n2 {
                for j in i..n2 {
                    keys[j+n1] = b.keys[j]+a.nrow*a.ncol;
                    data[j+n1] = b.data[j];
                }
            }
            else {
                let h:Simd<usize, LANES> = Simd::splat(a.nrow*a.ncol);
                let x:Simd<usize, LANES> = Simd::from_slice(&b.keys[i..i+LANES]);
                let y = x + h;
                
                Simd::copy_to_slice(y, &mut keys[i+n1..i+n1+LANES]);

                let x:Simd<f64, LANES> = Simd::from_slice(&b.data[i..i+LANES]);
                Simd::copy_to_slice(x, &mut data[i+n1..i+n1+LANES]);
            }
        }
    }

    return SparseMatrix::new(n, m, keys, data);
}

pub fn identity(n:usize) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    for i in 0..n {
        keys.push(i*(n+1));
        data.push(1.0);
    }

    return SparseMatrix::new(n, n, keys, data);
}

pub fn norm(a:&SparseMatrix) -> f64 {
    const LANES:usize = 64;
    let mut s = 0.0;
    let n = a.keys.len();

    for i in (0..n).step_by(LANES) {
        if i+LANES > n {
            for j in i..n {
                s += a.data[j]*a.data[j];
            }
        }
        else {
            let x:Simd<f64, LANES> = Simd::from_slice(&a.data[i..i+LANES]);
            let y = x*x;
            s += y.reduce_sum();
        }
    }

    return s.sqrt();
}

pub fn dot(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    let a_arc = Arc::new(a.clone());
    let b_arc = Arc::new(b.clone());

    let mut n = 0;
    let mut m = 0;

    if a.ncol == b.nrow {
        n = a.nrow;
        m = b.ncol;

        let n1 = a.keys.len();
        let hmap = Arc::new(Mutex::new(vec![0.0;n*m]));

        let mut handles = vec![];
        let q = (n1 as f64/4.0).ceil() as usize;

        for r in (0..n1).step_by(q) {
            let a_curr = Arc::clone(&a_arc);
            let b_curr = Arc::clone(&b_arc);
            let h_map = Arc::clone(&hmap);
            
            let handle = thread::spawn(move || {
                let mut i = r;
                let mut h = h_map.lock().unwrap();

                while i < min(r+q, n1)  {
                    let a_key = a_curr.keys[i];
                    let a_d = a_curr.data[i];
                    let a_row = a_key/a_curr.ncol;
                    let a_col = a_key % a_curr.ncol;

                    let mut j = binary_search_next(&b_curr.keys, a_col*b_curr.ncol);
                    
                    while j < b_curr.keys.len() {
                        let b_key = b_curr.keys[j];
                        let b_d = b_curr.data[j];
                        let b_row = b_key/b_curr.ncol;
                        let b_col = b_key % b_curr.ncol;

                        if b_row == a_col {
                            let nkey = a_row*m+b_col;
                            h[nkey] += a_d*b_d;
                        }
                        else {
                            break;
                        }

                        j += 1;
                    }

                    i += 1;
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let h = hmap.lock().unwrap();

        for i in 0..h.len() {
            if h[i].abs() > 1e-10 {
                keys.push(i);
                data.push(h[i]);
            }
        }
    }
    
    return SparseMatrix::new(n, m, keys, data);
}

pub fn convert_to_array(a:&SparseMatrix) -> Vec<f64> {
    let mut out:Vec<f64> = vec![0.0;a.nrow*a.ncol];

    for i in 0..a.keys.len() {
        let key = a.keys[i];
        let d = a.data[i];
        out[key] = d;
    }

    return out;
}

pub fn run() {
    let n = 500;
    let m = 500;
    let k = 0; //4*n*m/5;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];
    let mut b:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
        b[i] = normal.sample(&mut rng);
    }

    let uniform = Uniform::new(0, n*m);

    for _ in 0..k {
        let j = uniform.sample(&mut rng);
        a[j] = 0.0;
    }

    for _ in 0..k {
        let j = uniform.sample(&mut rng);
        b[j] = 0.0;
    }

    let c = SparseMatrix::create(n, m, &a);
    let d = SparseMatrix::create(m, n, &b);

    // let e = get_sub_mat(&c, 1, 3, 2, 4);
    // let f = convert_to_array(&e);

    // let g = add(&c, &d);
    // let h = convert_to_array(&g);

    // let j = mul(&c, &d);
    // let k = convert_to_array(&j);

    // let c1 = Arc::new(c);
    // let d1 = Arc::new(d);

    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    let l = dot(&c, &d);
    let end_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    println!("{:?}", end_time-start_time);
    // let m = convert_to_array(&l);

    // let q = transpose(&c);
    // let y = convert_to_array(&q);

    // println!("{:?}", a);
    // println!();
    // println!("{:?}", b);
    // println!();
    // println!("{:?}", f);
    // println!();
    // println!("{:?}", h);
    // println!();
    // println!("{:?}", k);
    // println!();
    // println!("{:?}", m);
    // println!();
    // println!("{:?}", y);
}



