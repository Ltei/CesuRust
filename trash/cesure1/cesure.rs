
use std::clone::Clone;

use utils::matrix;
use utils::matrix::Matrix;
use utils::matrix_math::p_mult;
use utils::matrix_math::row_concatenate;
use network::utils::activation;
use network::utils::activation::ActivationType;
use network::utils::feedforward_gate;
use network::utils::feedforward_gate::FeedforwardGate;


pub const INFOS_DIMENSION : usize = 3;
pub const CONTEXT_DIMENSION : usize = 5;
pub const OUTPUT_DIMENSION : usize = 2;


pub struct Cesure {
    pub output_gate : FeedforwardGate,
    pub forget_gate : FeedforwardGate,
    pub memory_gate : FeedforwardGate,
    pub input_gate : FeedforwardGate,
    pub infos : Matrix,
    pub context : Matrix,
}


pub fn new() -> Cesure {
    let infos_context_dimension = INFOS_DIMENSION + CONTEXT_DIMENSION;
    let infos_context_output_dimension = infos_context_dimension + OUTPUT_DIMENSION;
    return Cesure {
        output_gate : feedforward_gate::new_auto(infos_context_dimension, OUTPUT_DIMENSION, 4, activation::new(&ActivationType::Sigmoid)),
        forget_gate : feedforward_gate::new_auto(infos_context_output_dimension, CONTEXT_DIMENSION, 4, activation::new(&ActivationType::Sigmoid)),
        memory_gate : feedforward_gate::new_auto(infos_context_output_dimension, CONTEXT_DIMENSION, 4, activation::new(&ActivationType::Sigmoid)),
        input_gate : feedforward_gate::new_auto(infos_context_output_dimension, CONTEXT_DIMENSION, 4, activation::new(&ActivationType::Tanh)),
        infos : matrix::new_row(INFOS_DIMENSION),
        context : matrix::new_row(CONTEXT_DIMENSION),
    }
}



impl Clone for Cesure {
    fn clone(&self) -> Cesure {
        return Cesure {
            output_gate : self.output_gate.clone(),
            forget_gate : self.forget_gate.clone(),
            memory_gate : self.memory_gate.clone(),
            input_gate : self.input_gate.clone(),
            infos : self.infos.clone(),
            context : self.context.clone(),
        }
    }
    fn clone_from(&mut self, source: &Cesure) {
        self.output_gate.clone_from(&source.output_gate);
        self.forget_gate.clone_from(&source.forget_gate);
        self.memory_gate.clone_from(&source.memory_gate);
        self.input_gate.clone_from(&source.input_gate);
        self.infos.clone_from(&source.infos);
        self.context.clone_from(&source.context);
    }
}



#[allow(dead_code)]
impl Cesure {

    pub fn set_infos(&mut self, infos : &Matrix) {
        assert!(infos.is_row() && infos.len == INFOS_DIMENSION);
        self.infos.clone_from(infos);
    }

    pub fn reset_context(&mut self) {
        self.context.set_zero();
    }

    pub fn compute_next(&mut self) -> Matrix {
        let infos_context = row_concatenate(&self.infos, &self.context);
        let output = self.output_gate.compute(&infos_context);
        let infos_context_output = row_concatenate(&infos_context, &output);
        let forget_out = self.forget_gate.compute(&infos_context_output);
        let memory_out = self.memory_gate.compute(&infos_context_output);
        let input_out = self.input_gate.compute(&infos_context_output);
        self.context.p_mult(&forget_out);
        self.context.add(&p_mult(&memory_out,&input_out));

        output
    }

    pub fn input_next(&mut self, input : &Matrix) {
        let mut infos_context_output = row_concatenate(&self.infos, &self.context);
        infos_context_output.row_concatenate(&input);
        let forget_out = self.forget_gate.compute(&infos_context_output);
        let memory_out = self.memory_gate.compute(&infos_context_output);
        let input_out = self.input_gate.compute(&infos_context_output);
        self.context.p_mult(&forget_out);
        self.context.add(&p_mult(&memory_out,&input_out));
    }

}