#![allow(dead_code)]

use utils::matrix::Matrix;
use utils::math::sigmoid;
use utils::math::sigmoid_deriv;
use utils::math::tanh;
use utils::math::tanh_deriv;

use utils::traits::Parse;


pub const TYPE_SIGMOID : u8 = 0;
pub const TYPE_TANH : u8 = 1;


pub enum ActivationType {
    Sigmoid,
    Tanh,
}


pub struct Activation {
    pub act_type: u8,
    pub activate: fn(&Matrix) -> Matrix,
    pub derivate: fn(&Matrix) -> Matrix,
}



impl Clone for Activation {
    fn clone(&self) -> Activation {
        Activation {
            act_type : self.act_type,
            activate : self.activate,
            derivate : self.derivate,
        }
    }
    fn clone_from(&mut self, source: &Activation) {
        self.act_type = source.act_type;
        self.activate = source.activate;
        self.derivate = source.derivate;
    }
}
impl Parse for Activation {
    fn to_string(&self) -> String {
        return match self.act_type {
            TYPE_SIGMOID => "sigmoid".to_string(),
            TYPE_TANH => "tanh".to_string(),
            _ => panic!("Unknown activation type"),
        }
    }
    fn from_string(str : &str) -> Activation {
        match str {
            "sigmoid" => Activation::new(TYPE_SIGMOID),
            "tanh" => Activation::new(TYPE_TANH),
            _ => panic!("Unknown activation type"),
        }
    }
}



impl Activation {

    pub fn new(activation_type : u8) -> Activation {
        match activation_type {
            TYPE_SIGMOID => {
                return Activation {
                    act_type : TYPE_SIGMOID,
                    activate : matrix_sigmoid,
                    derivate : matrix_sigmoid_deriv,
                }
            }
            TYPE_TANH => {
                return Activation {
                    act_type : TYPE_TANH,
                    activate : matrix_tanh,
                    derivate : matrix_tanh_deriv,
                }
            }
            _ => panic!("Unknown activation type"),
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