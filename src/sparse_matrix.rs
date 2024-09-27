#![allow(dead_code)]
use std::collections::HashMap;
use rand_distr::{Distribution, Normal, Uniform};
use rand::thread_rng;

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

    let k_start = r_start*a.ncol + c_start;
    let k = binary_search_next(&a.keys, k_start);
    
    let n = r_end-r_start+1;
    let m = c_end-c_start+1;

    let mut h = k;

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
            let k = binary_search_next(&a.keys, k_start);
            h = k;
        }

        else {
            let k_start = (i+1)*a.ncol + c_start;
            let k = binary_search_next(&a.keys, k_start);
            h = k;
        }
    }

    return SparseMatrix::new(n, m, keys, data);
}

pub fn add(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    if a.nrow == b.nrow && a.ncol == b.ncol {
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

        return SparseMatrix::new(a.nrow, a.ncol, keys, data);
    }
    
    return SparseMatrix::new(0, 0, keys, data);
}

pub fn sub(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    if a.nrow == b.nrow && a.ncol == b.ncol {
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

        return SparseMatrix::new(a.nrow, a.ncol, keys, data);
    }
    
    return SparseMatrix::new(0, 0, keys, data);
}

pub fn mul(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    if a.nrow == b.nrow && a.ncol == b.ncol {
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

        return SparseMatrix::new(a.nrow, a.ncol, keys, data);
    }
    
    return SparseMatrix::new(0, 0, keys, data);
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

    if a.nrow >= b.nrow && a.ncol >= b.ncol {
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

        return SparseMatrix::new(a.nrow, a.ncol, keys, data);
    }
    
    return SparseMatrix::new(0, 0, keys, data);
}

pub fn vstack(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();

    if a.ncol == b.ncol {
        for i in 0..a.keys.len() {
            keys.push(a.keys[i]);
            data.push(a.data[i]);
        }
    
        for i in 0..b.keys.len() {
            keys.push(b.keys[i]+a.nrow*a.ncol);
            data.push(b.data[i]);
        }

        return SparseMatrix::new(a.nrow+b.nrow, a.ncol, keys, data);
    }

    return SparseMatrix::new(0, 0, keys, data);
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
    let mut out:f64 = 0.0;

    for i in 0..a.data.len() {
        out += a.data[i]*a.data[i];
    }
    return out.sqrt();
}

pub fn dot(a:&SparseMatrix, b:&SparseMatrix) -> SparseMatrix {
    let mut keys:Vec<usize> = Vec::new();
    let mut data:Vec<f64> = Vec::new();
    let mut hmap:HashMap<usize, f64> = HashMap::new();

    if a.ncol == b.nrow {
        let n = a.nrow;
        let m = b.ncol;

        for i in 0..a.keys.len() {
            let a_key = a.keys[i];
            let a_d = a.data[i];
            let a_row = a_key/a.ncol;
            let a_col = a_key % a.ncol;
            
            let k = binary_search_next(&b.keys, a_col*b.ncol);
            for j in k..b.keys.len() {
                let b_key = b.keys[j];
                let b_d = b.data[j];
                let b_row = b_key/b.ncol;
                let b_col = b_key % b.ncol;

                if b_row == a_col {
                    let nkey = a_row*m+b_col;
                    let u = hmap.entry(nkey).or_insert(0.0);
                    *u += a_d*b_d;
                }
                else {
                    break;
                }
            }
        }

        for k in hmap.keys() {
            keys.push(*k);
        }

        keys.sort();

        for k in &keys {
            data.push(*hmap.get(k).unwrap());
        }

        return SparseMatrix::new(n, m, keys, data);
    }
    
    return SparseMatrix::new(0, 0, keys, data);
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
    let n = 1000;
    let m = 1000;

    let mut rng = thread_rng();
    let normal:Normal<f64> = Normal::new(0.0, 1.0).ok().unwrap();
    
    let mut a:Vec<f64> = vec![0.0;n*m];
    let mut b:Vec<f64> = vec![0.0;n*m];

    for i in 0..n*m {
        a[i] = normal.sample(&mut rng);
        b[i] = normal.sample(&mut rng);
    }

    let uniform = Uniform::new(0, n*m);

    for _ in 0..999000 {
        let j = uniform.sample(&mut rng);
        a[j] = 0.0;
    }

    for _ in 0..999000 {
        let j = uniform.sample(&mut rng);
        b[j] = 0.0;
    }

    let c = SparseMatrix::create(n, m, &a);
    let d = SparseMatrix::create(m, n, &b);

    let e = get_sub_mat(&c, 1, 3, 2, 4);
    let f = convert_to_array(&e);

    let g = add(&c, &d);
    let h = convert_to_array(&g);

    let j = mul(&c, &d);
    let k = convert_to_array(&j);

    let l = dot(&c, &d);
    let m = convert_to_array(&l);

    let q = transpose(&c);
    let y = convert_to_array(&q);

    println!("{:?}", a);
    println!();
    println!("{:?}", b);
    println!();
    println!("{:?}", f);
    println!();
    println!("{:?}", h);
    println!();
    println!("{:?}", k);
    println!();
    println!("{:?}", m);
    println!();
    println!("{:?}", y);
}



