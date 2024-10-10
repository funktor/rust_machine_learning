#![allow(dead_code)]
use crate::matrix_utils::*;
use std::cmp::min;
use rand::Rng;
use rand_distr::StandardNormal;
use std::thread;
use std::sync::{Arc, Mutex};

pub fn predict(data: &[f64], weights: &[f64], bias: f64) -> f64 {
    let sum: f64 = dot_product_simd(&weights, &data);
    return sum + bias;
}

#[derive(Clone)]
pub struct LinearRegression {
    weights: Arc<Mutex<Vec<f64>>>,
    bias: f64,
    num_epochs: usize,
    batch_size: usize,
    l1_reg: f64,
    l2_reg: f64,
    fixed_learning_rate: f64,
}

impl LinearRegression {
    pub fn new(
        &n: &usize,
        &num_epochs:&usize, 
        &batch_size:&usize, 
        &l1_reg:&f64, 
        &l2_reg:&f64, 
        &fixed_learning_rate:&f64
    ) -> Self {

        let mut weights: Vec<f64> = vec![0.0;n];
        for i in 0..n {
            weights[i] = rand::thread_rng().sample(StandardNormal);
        }

        Self {
            weights:Arc::new(Mutex::new(weights)),
            bias:0.0,
            num_epochs,
            batch_size,
            l1_reg,
            l2_reg,
            fixed_learning_rate,
        }
    }
}

impl LinearRegression {
    pub fn predict(&self, data:&Vec<f64>) -> f64 {
        return predict(
            &data, 
            &self.weights.lock().unwrap(), 
            self.bias
        );
    }
}

impl LinearRegression {
    pub fn get_errors(
        &self, 
        data:Arc<Vec<f64>>, 
        labels:Arc<Vec<f64>>,
        n:usize,
        m:usize,
    ) -> Vec<f64> {

        let mut errors:Vec<f64> = vec![0.0;n];

        let mut handles = vec![];
        let q = (n as f64/4.0).ceil() as usize;

        for i in (0..n).step_by(q) {
            let d = Arc::clone(&data);
            let w = Arc::clone(&self.weights);
            let b = Arc::new(self.bias);
            let u = Arc::clone(&labels);
            
            let handle = thread::spawn(move || {
                let mut results = vec![];
                for j in i..min(i+q, n)  {
                    let pred = 
                        predict(
                            &d[j*m..min((j+1)*m, n*m)], 
                            &w.lock().unwrap(), 
                            *b);
                    results.push((pred-u[j], j));
                }

                return results;
            });

            handles.push(handle);
        }

        for handle in handles {
            let err = handle.join().unwrap();
            for z_err in &err {
                errors[z_err.1] = z_err.0;
            }
        }

        return errors;
    }
}

impl LinearRegression {
    pub fn loss(
        &self, 
        data:Arc<Vec<f64>>,  
        labels:Arc<Vec<f64>>,
        n:usize,
        m:usize,
    ) -> f64 {  

        let mut loss: f64 = 0.0;

        let errors = 
            self.get_errors(
                Arc::clone(&data), 
                Arc::clone(&labels),
                n, 
                m
            );
        
        for err in errors {
            loss += (1.0/n as f64)*err*err;
        }

        let w = self.weights.lock().unwrap();

        for i in 0..m {
            loss += self.l2_reg*w[i]*w[i];
            loss += self.l1_reg*w[i].abs();
        }

        return loss + self.l1_reg*self.bias.abs() + self.l2_reg*self.bias*self.bias;
    }
}

impl LinearRegression {
    pub fn get_weights_gradient(
        &self, 
        data:Arc<Vec<f64>>, 
        errors:&Vec<f64>,
        n:usize,
        m:usize,
    ) -> Vec<f64> {

        let mut gradients:Vec<f64> = vec![0.0;m];
        let w = self.weights.lock().unwrap();
        let res = matrix_multiply_simd(&errors, &data, 1, n, m);
        
        for i in 0..m {
            let sum = 2.0*res[i];
            let mut gradient:f64 = 1.0/n as f64*sum + 2.0*self.l2_reg*w[i];

            if w[i] > 0.0 {
                gradient += self.l1_reg;
            }
            else if w[i] < 0.0 {
                gradient += -self.l1_reg;
            }

            gradients[i] = gradient;
        }

        return gradients;
    }
}

impl LinearRegression {
    pub fn gradient_descent(
        &mut self, 
        data:Arc<Vec<f64>>, 
        labels:Arc<Vec<f64>>,
        n:usize,
        m:usize,
    ) {

        let errors = 
            self.get_errors(
                Arc::clone(&data), 
                Arc::clone(&labels),
                n, 
                m
            );
        
        let gradients = 
            self.get_weights_gradient(
                Arc::clone(&data), 
                &errors,
                n, 
                m
            );

        let mut w = self.weights.lock().unwrap();

        for i in 0..m {
            w[i] -= self.fixed_learning_rate*gradients[i];
        }
        
        let mut sum: f64 = 0.0;
        for j in 0..n {
            sum += 2.0*errors[j];
        }

        let mut gradient:f64 = 1.0/n as f64*sum + 2.0*self.l2_reg*self.bias;
        
        if self.bias > 0.0 {
            gradient += self.l1_reg;
        }
        else if self.bias < 0.0 {
            gradient += -self.l1_reg;
        }

        self.bias -= self.fixed_learning_rate*gradient;

    }
}

impl LinearRegression {
    pub fn train(
        &mut self, 
        data:&Vec<f64>, 
        labels:&Vec<f64>,
        n:usize,
        m:usize,
    ) {
        let d = Arc::new(data.clone());
        let u = Arc::new(labels.clone());

        let mut batched_data:Vec<(Arc<Vec<f64>>, Arc<Vec<f64>>)> = Vec::new();

        for i in (0..n).step_by(self.batch_size) {
            let x = &d[i*m..min((i+self.batch_size)*m, n*m)];
            let y = &u[i..min(i+self.batch_size, n)];
            batched_data.push((Arc::new(x.to_vec()), Arc::new(y.to_vec())));
        }

        for epoch in 1..self.num_epochs+1 {
            for i in 0..batched_data.len() {
                let batch = &batched_data[i];
                let n1 = batch.1.len();
                let d_e = Arc::clone(&batch.0);
                let u_e = Arc::clone(&batch.1);
                self.gradient_descent(d_e, u_e, n1, m);
            }

            let loss = self.loss(Arc::clone(&d), Arc::clone(&u), n, m);
            println!("Epoch = {:?}, Loss = {:?}", epoch, loss);
        }
    }
}

pub fn run() {
    let n = 1000;
    let m = 3000;

    let mut lr = LinearRegression::new(&m, &1000, &1024, &0.0, &0.001, &0.005);
    let mut data:Vec<f64> = vec![0.0;n*m];
    let mut labels:Vec<f64> = vec![0.0;n];

    for i in 0..n {
        for j in 0..m {
            data[i*m + j] = rand::thread_rng().sample(StandardNormal);
        }
        labels[i] = rand::thread_rng().sample(StandardNormal);
    }

    lr.train(&data, &labels, n, m);

    let mut query:Vec<f64> = vec![0.0;m];
    for j in 0..m {
        query[j] = rand::thread_rng().sample(StandardNormal);
    }

    let preds = lr.predict(&data);
    println!("{:?}", preds);
}