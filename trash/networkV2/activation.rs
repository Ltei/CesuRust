#![allow(dead_code)]

use utils::math::sigmoid;
use utils::math::sigmoid_deriv;
use utils::math::tanh;
use utils::math::tanh_deriv;
use utils::matrix::Matrix;
use utils::traits::Parse;



pub enum ActivationType {
    Sigmoid,
    Tanh,
}


pub struct Activation {
    pub act_type: ActivationType,
    pub activate: fn(&Matrix) -> Matrix,
    pub derivate: fn(&Matrix) -> Matrix,
}



impl Clone for Activation {
    fn clone(&self) -> Activation {
        match self.act_type {
            ActivationType::Sigmoid => return Activation::new(&ActivationType::Sigmoid),
            ActivationType::Tanh => return Activation::new(&ActivationType::Tanh),
        }
    }
    fn clone_from(&mut self, source: &Activation) {
        match source.act_type {
            ActivationType::Sigmoid => self.act_type = ActivationType::Sigmoid,
            ActivationType::Tanh => self.act_type = ActivationType::Tanh,
        }
        self.activate = source.activate;
        self.derivate = source.derivate;
    }
}
impl Parse for Activation {
    fn to_string(&self) -> String {
        return match self.act_type {
            ActivationType::Sigmoid => "sigmoid".to_string(),
            ActivationType::Tanh => "tanh".to_string(),
        }
    }
    fn from_string(str : &str) -> Activation {
        if str.eq("tanh") {
            return Activation::new(&ActivationType::Tanh);
        } else if str.eq("sigmoid") {
            return Activation::new(&ActivationType::Sigmoid);
        } else {
            panic!();
        }
    }
}



impl Activation {

    pub fn new(act_type : &ActivationType) -> Activation {
        match act_type {
            &ActivationType::Sigmoid => {
                return Activation {
                    act_type : ActivationType::Sigmoid,
                    activate : matrix_sigmoid,
                    derivate : matrix_sigmoid_deriv,
                }
            }
            &ActivationType::Tanh => {
                return Activation {
                    act_type : ActivationType::Tanh,
                    activate : matrix_tanh,
                    derivate : matrix_tanh_deriv,
                }
            }
        }
    }

}



#[inline]
fn matrix_sigmoid(matrix : & Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, matrix.cols);
    for index in 0..matrix.len {
        result.datas[index] = sigmoid(matrix.datas[index]);
    }
    return result
}
#[inline]
fn matrix_sigmoid_deriv(matrix : & Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, matrix.cols);
    for index in 0..matrix.len {
        result.datas[index] = sigmoid_deriv(matrix.datas[index]);
    }
    return result
}
#[inline]
fn matrix_tanh(matrix : & Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, matrix.cols);
    for index in 0..matrix.len {
        result.datas[index] = tanh(matrix.datas[index]);
    }
    return result
}
#[inline]
fn matrix_tanh_deriv(matrix : & Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, matrix.cols);
    for index in 0..matrix.len {
        result.datas[index] = tanh_deriv(matrix.datas[index]);
    }
    return result
}