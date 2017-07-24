#![allow(dead_code)]

extern crate rand;

use std::ops::Add;

use network::gate::activation::Activation;

use utils::matrix::Matrix;
use utils::matrix_math::m_dot;
use utils::matrix_math::p_mult;
use utils::matrix_math::mult_scl;
use utils::matrix_math::transpose;
use utils::matrix_math::row_append;
use utils::traits::Parse;



pub struct FeedforwardGate {
    pub input_dimension : usize,
    pub output_dimension : usize,
    pub nb_layers : usize,
    pub layers : Vec<Matrix>,
    pub activation : Activation,
}

pub struct VerboseOutput {
    pub input_bias : Matrix,
    pub outputs_unact : Vec<Matrix>,
    pub outputs_act_bias : Vec<Matrix>,
    pub output : Matrix,
}

type WeightsChanges = Vec<Matrix>;



impl Clone for FeedforwardGate {
    fn clone(&self) -> FeedforwardGate {
        let mut output = FeedforwardGate {
            input_dimension : self.input_dimension,
            output_dimension : self.output_dimension,
            nb_layers : self.nb_layers,
            layers : Vec::with_capacity(self.nb_layers),
            activation : self.activation.clone(),
        };
        for i in 0..self.nb_layers {
            output.layers.push(self.layers[i].clone());
        }
        return output;
    }
    fn clone_from(&mut self, source: &FeedforwardGate) {
        self.input_dimension = source.input_dimension;
        self.output_dimension = source.output_dimension;
        self.nb_layers = source.nb_layers;
        self.layers = Vec::with_capacity(source.nb_layers);
        for i in 0..self.nb_layers {
            self.layers.push(source.layers[i].clone());
        }
        self.activation.clone_from(&source.activation);
    }
}
impl Parse for FeedforwardGate {
    fn to_string(&self) -> String {
        let mut output = format!("{} {} {} {}", self.input_dimension, self.output_dimension, self.nb_layers, self.activation.to_string());
        for i in 0..self.nb_layers {
            output = output.add(&format!("\n{}", &self.layers[i].to_string()));
        }
        return output;
    }
    fn from_string(str : &str) -> FeedforwardGate {
        let lines : Vec<&str> = str.split("\n").collect();
        assert!(lines.len() >= 2); // header + at least one layer

        let header : Vec<&str> = lines[0].split(" ").collect();
        assert!(header.len() == 4);
        let input_dimension = header[0].parse().unwrap();
        let output_dimension = header[1].parse().unwrap();
        let nb_layers = header[2].parse().unwrap();
        let activation = Activation::from_string(header[3]);

        assert!(lines.len() == nb_layers + 1); // header + layers
        let mut layers = Vec::with_capacity(nb_layers);
        for i in 1..lines.len() {
            layers.push( Matrix::from_string(lines[i]) );
        }

        return FeedforwardGate {
            input_dimension : input_dimension,
            output_dimension : output_dimension,
            nb_layers : nb_layers,
            layers : layers,
            activation : activation,
        }
    }
}



impl FeedforwardGate {

    /**
    * Create a new FeedforwardGate object
    * @input input_dimension : The gate input dimension
    * @input output_dimension : The gate output dimension
    * @input hiddens_dimensions : The hidden layers' dimension (The number of layers
    *                             will be hiddens_dimensions.len()+1)
    * @input activation : The activation function
    * @return The newly created FeedforwardGate object
    */
    pub fn new(input_dimension : usize, output_dimension : usize, hiddens_dimensions : Vec<usize>, activation : Activation) -> FeedforwardGate {
        assert!(input_dimension > 0 && output_dimension > 0);
        let nb_layers = hiddens_dimensions.len() + 1;
        let mut output = FeedforwardGate {
            input_dimension : input_dimension,
            output_dimension : output_dimension,
            nb_layers : nb_layers,
            layers : Vec::with_capacity(nb_layers),
            activation : activation,
        };
        if nb_layers == 1 {
            output.layers.push(Matrix::new(input_dimension+1, output_dimension));
        } else {
            output.layers.push(Matrix::new(input_dimension+1, hiddens_dimensions[0]));
            for i in 0..(nb_layers-2) {
                output.layers.push(Matrix::new(hiddens_dimensions[i]+1, hiddens_dimensions[i+1]));
            }
            output.layers.push(Matrix::new(hiddens_dimensions[nb_layers-2]+1, output_dimension));
        }
        output.weight_init_xavier(&mut rand::thread_rng());
        return output;
    }

    /**
    * Create a new FeedforwardGate object with automatically determined number of  hidden layers' neurons
    * @input input_dimension : The gate input dimension
    * @input output_dimension : The gate output dimension
    * @input nb_layers : The number of layers
    * @input activation : The activation function
    * @return The newly created FeedforwardGate object
    */
    pub fn new_auto(input_dimension : usize, output_dimension : usize, nb_layers : usize, activation : Activation) -> FeedforwardGate {
        assert!(input_dimension > 0 && output_dimension > 0 && nb_layers > 0);
        let mut output = FeedforwardGate {
            input_dimension : input_dimension,
            output_dimension : output_dimension,
            nb_layers : nb_layers,
            layers : Vec::with_capacity(nb_layers),
            activation : activation,
        };

        let in_dim = input_dimension as f64;
        let out_dim = output_dimension as f64;
        let x_div = nb_layers as f64;
        let mut last_out_dimension = input_dimension;

        for i in 0..nb_layers {
            let x : f64 = (i+1) as f64 / x_div;
            let new_out_dimension = x * out_dim + (1.0-x) * in_dim;
            let mut new_out_dimension = new_out_dimension.round() as usize;
            if i < nb_layers-1 {
                new_out_dimension += 1;
            }
            output.layers.push(Matrix::new(last_out_dimension+1, new_out_dimension));
            last_out_dimension = new_out_dimension;
        }
        output.weight_init_xavier(&mut rand::thread_rng());
        return output;
    }

    /**
    * Initialize the weights using XAVIER initialization
    * @input rand : The ThreadRng object to use for random
    */
    fn weight_init_xavier(&mut self, mut rand : &mut rand::ThreadRng) {
        for i in 0..self.nb_layers {
            self.layers[i].set_random(-1.0, 1.0, rand); //TODO
        }
    }

    /**
    * @input magnitude : The amount of random
    * @input rand : The ThreadRng object to use for random
    * @return The cloned FeedforwardGate
    */
    pub fn clone_randomized(&self, magnitude : f64, rand : &mut rand::ThreadRng) -> FeedforwardGate {
        let mut layers = Vec::with_capacity(self.nb_layers);
        for i in 0..self.nb_layers {
            layers.push( self.layers[i].clone_randomized(magnitude,rand) );
        }
        let activation = self.activation.clone();
        return FeedforwardGate {
            input_dimension : self.input_dimension,
            output_dimension : self.output_dimension,
            nb_layers : self.nb_layers,
            layers : layers,
            activation : activation,
        }
    }

    /**
    * Print all the layers
    */
    pub fn print(&self) {
        for i in 0..self.nb_layers {
            println!("Layer #{} :", i);
            self.layers[i].print();
        }
    }

    /**
    * Return the number of neurons in this network
    * @return The number of neurons
    */
    pub fn get_nb_neurons(&self) -> usize {
        let mut nb_neurons = 0;
        for layer in &self.layers {
            nb_neurons += layer.len;
        }
        nb_neurons
    }

    /**
    * Compute an input and return the computed output
    * @input input : The input to compute
    * @return The computed output
    */
    pub fn compute(&self, input : &Matrix) -> Matrix {
        assert!(input.is_row() && input.len == self.input_dimension);
        let mut output = row_append(input, 1.0);
        for i in 0..self.nb_layers {
            output.m_dot(&self.layers[i]);
            output = (self.activation.activate)(&output);
            if i < self.nb_layers-1 {
                output.row_append(1.0);
            }
        }
        return output;
    }

    /**
    * Compute an input and return a VerboseOutput object
    * containing all the layers' outputs
    * @input input : The input to compute
    * @return The VerboseOutput object
    */
    pub fn compute_verbose(&self, input : &Matrix) -> VerboseOutput {
        assert!(input.is_row() && input.len == self.input_dimension);

        let mut outputs_unact = Vec::with_capacity(self.nb_layers);
        let mut outputs_act_bias = Vec::with_capacity(self.nb_layers);

        let input_bias = row_append(&input, 1.0);

        outputs_unact.push( m_dot(&input_bias, &self.layers[0]) );
        let mut output_act_bias : Matrix = (self.activation.activate)(&outputs_unact[0]);
        output_act_bias.row_append(1.0);
        outputs_act_bias.push(output_act_bias);
        for i in 1..self.nb_layers {
            outputs_unact.push( m_dot(&outputs_act_bias[i-1], &self.layers[i]) );
            let mut output_act_bias : Matrix = (self.activation.activate)(&outputs_unact[i]);
            output_act_bias.row_append(1.0);
            outputs_act_bias.push(output_act_bias);
        }

        let output = (self.activation.activate)(&outputs_unact[self.nb_layers-1]);

        assert!(outputs_unact.len() == self.nb_layers);
        assert!(outputs_act_bias.len() == self.nb_layers);


        return VerboseOutput {
            input_bias : input_bias,
            outputs_unact : outputs_unact,
            outputs_act_bias : outputs_act_bias,
            output : output,
        }
    }

    /**
    * Backpropagate an error signal to change the weights
    * The final output error signal is = (output-ideal)
    * @input output_v : The la computation's VerboseOutput object
    * @input signal : The output error signal
    * @input last_changes : A LastChanges object with the last weights changes
    *                       put None if it's the first iteration
    * @input momentum : The momentum
    * @return A tuple composed by the input error signal and the last changes
    */
    pub fn backpropagate(&mut self, output_v : &VerboseOutput, signal : &Matrix, learning_rate : f64,
                                                    last_changes : Option<WeightsChanges>, momentum : f64) -> (Matrix,WeightsChanges) {
        assert!(signal.len == self.output_dimension);

        let mut hidden_signals = Vec::with_capacity(self.nb_layers);
        let mut weights_deltas = Vec::with_capacity(self.nb_layers);

        for _ in 0..self.nb_layers {
            hidden_signals.push(Matrix::new(1,1));
            weights_deltas.push(Matrix::new(1,1));
        }

        if self.nb_layers == 1 {
            hidden_signals[0] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[0]) );
            weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
            weights_deltas[0].mult_scl(-1.0 * learning_rate);

        } else { // self.nb_layers >= 2
            hidden_signals[self.nb_layers-1] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[self.nb_layers-1]) );
            weights_deltas[self.nb_layers-1] = m_dot( &transpose(&output_v.outputs_act_bias[self.nb_layers-2]), &hidden_signals[self.nb_layers-1] );
            weights_deltas[self.nb_layers-1].mult_scl(-1.0 * learning_rate);

            let mut layer_i = self.nb_layers-2;
            while layer_i >= 1 {
                let mut hidden_signal = m_dot(&hidden_signals[layer_i+1], &transpose(&self.layers[layer_i+1]));
                hidden_signal.delete_last_col();
                hidden_signal.p_mult(&(self.activation.derivate)(&output_v.outputs_unact[layer_i]));

                hidden_signals[layer_i] = hidden_signal;
                weights_deltas[layer_i] = m_dot( &transpose(&output_v.outputs_act_bias[layer_i-1]), &hidden_signals[layer_i] );
                weights_deltas[layer_i].mult_scl(-1.0 * learning_rate);
                layer_i -= 1;
            }

            let mut hidden_signal = m_dot(&hidden_signals[1], &transpose(&self.layers[1]));
            hidden_signal.delete_last_col();
            hidden_signal.p_mult( &(self.activation.derivate)(&output_v.outputs_unact[0]) );

            hidden_signals[0] = hidden_signal;
            weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
            weights_deltas[0].mult_scl(-1.0 * learning_rate);
        }



        match last_changes {
            Some(mut changes) => {
                for layer_i in 0..self.nb_layers {
                    weights_deltas[layer_i].add( changes[layer_i].mult_scl(momentum) );
                }
            }
            None => {}
        }

        for layer_i in 0..self.nb_layers {
            self.layers[layer_i].add(&weights_deltas[layer_i]);
        }

        let mut input_signal = m_dot(&hidden_signals[0], &transpose(&self.layers[0]));
        input_signal.delete_last_col();

        assert!(input_signal.is_finite());
        return (input_signal, weights_deltas);
    }

    /**
    * Backpropagate an error signal to get the weights changes
    * It won't apply the weights changes, use apply_changes() if you want to apply them
    * @input output_v : The la computation's VerboseOutput object
    * @input signal : The output error signal
    * @input last_changes : A LastChanges object with the last weights changes
    *                       put None if it's the first iteration
    * @input momentum : The momentum
    * @return A tuple composed by the input error signal and the last changes
    */
    pub fn backpropagate_no_change(&mut self, output_v : &VerboseOutput, signal : &Matrix, learning_rate : f64,
                                                    last_changes : &Option<WeightsChanges>, momentum : f64) -> (Matrix,WeightsChanges) {
        assert!(signal.len == self.output_dimension);

        let mut hidden_signals = Vec::with_capacity(self.nb_layers);
        let mut weights_deltas = Vec::with_capacity(self.nb_layers);

        for _ in 0..self.nb_layers {
            hidden_signals.push(Matrix::new(1,1));
            weights_deltas.push(Matrix::new(1,1));
        }

        if self.nb_layers == 1 {
            hidden_signals[0] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[0]) );
            weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
            weights_deltas[0].mult_scl(-1.0 * learning_rate);

        } else { // self.nb_layers >= 2
            hidden_signals[self.nb_layers-1] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[self.nb_layers-1]) );
            weights_deltas[self.nb_layers-1] = m_dot( &transpose(&output_v.outputs_act_bias[self.nb_layers-2]), &hidden_signals[self.nb_layers-1] );
            weights_deltas[self.nb_layers-1].mult_scl(-1.0 * learning_rate);

            let mut layer_i = self.nb_layers-2;
            while layer_i >= 1 {
                let mut hidden_signal = m_dot(&hidden_signals[layer_i+1], &transpose(&self.layers[layer_i+1]));
                hidden_signal.delete_last_col();
                hidden_signal.p_mult(&(self.activation.derivate)(&output_v.outputs_unact[layer_i]));

                hidden_signals[layer_i] = hidden_signal;
                weights_deltas[layer_i] = m_dot( &transpose(&output_v.outputs_act_bias[layer_i-1]), &hidden_signals[layer_i] );
                weights_deltas[layer_i].mult_scl(-1.0 * learning_rate);
                layer_i -= 1;
            }

            let mut hidden_signal = m_dot(&hidden_signals[1], &transpose(&self.layers[1]));
            hidden_signal.delete_last_col();
            hidden_signal.p_mult( &(self.activation.derivate)(&output_v.outputs_unact[0]) );

            hidden_signals[0] = hidden_signal;
            weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
            weights_deltas[0].mult_scl(-1.0 * learning_rate);
        }



        match last_changes {
            &Some(ref last_changes) => {
                for layer_i in 0..self.nb_layers {
                    weights_deltas[layer_i].add( &mult_scl(&last_changes[layer_i], momentum) );
                }
            }
            &None => {}
        }

        let mut input_signal = m_dot(&hidden_signals[0], &transpose(&self.layers[0]));
        input_signal.delete_last_col();

        assert!(input_signal.is_finite());
        return (input_signal, weights_deltas);
    }

    /**
    * Apply the weights changes to the layers
    * @input weights_changes : The weights changes to apply
    */
    pub fn apply_changes(&mut self, weights_changes : &WeightsChanges) {
        assert!(weights_changes.len() == self.layers.len());
        for i in 0..self.layers.len() {
            self.layers[i].add(&weights_changes[i]);
        }
    }

}



/*/**
    * Create a new FeedforwardGate object
    * @input input_dimension : The gate input dimension
    * @input output_dimension : The gate output dimension
    * @input hidden_dimension : The hidden layers' output dimension
    * @input nb_layers : The number of layers
    * @input activation : The activation function
    * @return The newly created FeedforwardGate object
    */
pub fn new(input_dimension : usize, output_dimension : usize, hidden_dimension : usize, nb_layers : usize, activation : Activation) -> FeedforwardGate {
    assert!(input_dimension > 0 && output_dimension > 0 && hidden_dimension > 0 && nb_layers > 0);
    let mut output = FeedforwardGate {
        input_dimension : input_dimension,
        output_dimension : output_dimension,
        nb_layers : nb_layers,
        layers : Vec::with_capacity(nb_layers),
        activation : activation,
    };
    if nb_layers == 1 {
        output.layers.push(Matrix::new(input_dimension+1, output_dimension));
    } else {
        output.layers.push(Matrix::new(input_dimension+1, hidden_dimension));
        for _ in 1..(nb_layers-1) {
            output.layers.push(Matrix::new(hidden_dimension+1, hidden_dimension));
        }
        output.layers.push(Matrix::new(hidden_dimension+1, output_dimension));
    }
    output.weight_init_xavier(&mut rand::thread_rng());
    return output;
}*/
/*/**
    * Backpropagate an error signal to change the weights
    * The final output error signal is = (output-ideal)
    * @input output_v : The la computation's VerboseOutput object
    * @input signal : The output error signal
    * @output The input error signal
    */
pub fn backpropagate_error_signal(&mut self, output_v : &VerboseOutput, signal : &Matrix, learning_rate : f64) -> Matrix {
    assert!(signal.len == self.output_dimension);

    let mut hidden_signals = Vec::with_capacity(self.nb_layers);
    let mut weights_deltas = Vec::with_capacity(self.nb_layers);


    for _ in 0..self.nb_layers {
        hidden_signals.push(Matrix::new(1,1));
        weights_deltas.push(Matrix::new(1,1));
    }

    if self.nb_layers == 1 {
        hidden_signals[0] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[0]) );
        weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
        weights_deltas[0].mult_scl(-1.0 * learning_rate);

    } else { // self.nb_layers >= 2
        hidden_signals[self.nb_layers-1] = p_mult( signal, &(self.activation.derivate)(&output_v.outputs_unact[self.nb_layers-1]) );
        weights_deltas[self.nb_layers-1] = m_dot( &transpose(&output_v.outputs_act_bias[self.nb_layers-2]), &hidden_signals[self.nb_layers-1] );
        weights_deltas[self.nb_layers-1].mult_scl(-1.0 * learning_rate);

        let mut layer_i = self.nb_layers-2;
        while layer_i >= 1 {
            let mut hidden_signal = m_dot(&hidden_signals[layer_i+1], &transpose(&self.layers[layer_i+1]));
            hidden_signal.delete_last_col();
            hidden_signal.p_mult(&(self.activation.derivate)(&output_v.outputs_unact[layer_i]));

            hidden_signals[layer_i] = hidden_signal;
            weights_deltas[layer_i] = m_dot( &transpose(&output_v.outputs_act_bias[layer_i-1]), &hidden_signals[layer_i] );
            weights_deltas[layer_i].mult_scl(-1.0 * learning_rate);
            layer_i -= 1;
        }

        let mut hidden_signal = m_dot(&hidden_signals[1], &transpose(&self.layers[1]));
        hidden_signal.delete_last_col();
        hidden_signal.p_mult( &(self.activation.derivate)(&output_v.outputs_unact[0]) );

        hidden_signals[0] = hidden_signal;
        weights_deltas[0] = m_dot( &transpose(&output_v.input_bias), &hidden_signals[0] );
        weights_deltas[0].mult_scl(-1.0 * learning_rate);
    }




    for layer_i in 0..self.nb_layers {
        self.layers[layer_i].add(&weights_deltas[layer_i]);
    }

    let mut hidden_signal = m_dot(&hidden_signals[0], &transpose(&self.layers[0]));
    hidden_signal.delete_last_col();
    return hidden_signal;
}*/