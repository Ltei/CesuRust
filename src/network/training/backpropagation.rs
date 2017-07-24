
use std::f64;

use network::cesure::Cesure;
use network::training::training_set::TrainingSet;
use network::training::error_calculation::ErrorCalculation;

use network::music::INFOS_DIMENSION;
use network::music::CHORD_DIMENSION;

use utils::matrix::Matrix;
use utils::traits::Parse;
use utils::io::{AsyncStdinReader, AsyncStdinRead};


pub fn train(cesure: &mut Cesure, training_sets: &Vec<TrainingSet>, error_calculation: &ErrorCalculation, learning_rate: f64, momentum: f64, iterations: usize) {

    let mut stdin = AsyncStdinReader::new();
    let mut learning_rate = learning_rate;
    let mut momentum = momentum;
    let mut iterations = iterations;
    let mut show = true;

    for iteration in 0..iterations {

        if let Some(line) = stdin.read_line() {
            let mut args = line.as_str().split_whitespace();
            match args.next() {
                Some(arg) => {
                    match arg {
                        "stop" => { break; },
                        "show" => { show = true; },
                        "hide" => { show = false; },
                        "setlr" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(lr) => {
                                            match lr < 0.0 {
                                                true => { println!("Learning rate has to be positive"); },
                                                false => { learning_rate = lr; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setlr"); }
                            }
                        }
                        "setmom" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mom) => {
                                            match mom < 0.0 {
                                                true => { println!("Momentum has to be positive"); },
                                                false => { momentum = mom; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmom"); }
                            }
                        }
                        /*"setiters" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(iters) => {
                                            iterations = iters;
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setiters"); }
                            }
                        }*/
                        _ => println!("Unknown command [{}]", line),
                    }
                },
                None => {}
            }
        }

        let mut error_sum = 0.0;
        let mut output_gate_changes_sum = Vec::with_capacity(cesure.output_gate.nb_layers);
        let mut output_gate_nb_changes = 0.0;
        let mut memory_gate_changes_sum = Vec::with_capacity(cesure.memory_gate.nb_layers);
        let mut memory_gate_nb_changes = 0.0;

        for set_i in 0..training_sets.len() {
            let training_set = &training_sets[set_i];
            let sequence_len = training_set.compute_sequence.len();

            cesure.new_sequence(&training_set.infos);

            for injection in &training_set.inject_sequence {
                cesure.inject_next(injection);
            }

            let mut errors = Vec::with_capacity(sequence_len);
            let mut outputs = Vec::with_capacity(sequence_len);

            for i in 0..sequence_len {
                let output = cesure.compute_next_verbose();
                let error = (error_calculation.calculate)(&output.output_out.output, &training_set.compute_sequence[i]); //sub(&output.output_out.output, &training_set.compute_sequence[i]);
                error_sum += error.get_abs_avg();
                outputs.push(output);
                errors.push(error);
            }

            let mut output_gate_last_changes = None;
            let mut memory_gate_last_changes = None;
            let mut memory_gate_signal : Option<Matrix> = None;

            for i in 0..sequence_len {
                let i = sequence_len-1 - i;

                let (signal, weights_changes) = cesure.output_gate.backpropagate_no_change(&outputs[i].output_out, &errors[i], learning_rate,
                                                                     &output_gate_last_changes, momentum);
                weights_changes_add_or_clone(&mut output_gate_changes_sum, &weights_changes);
                output_gate_nb_changes += 1.0;
                output_gate_last_changes = Some(weights_changes);

                if i > 1 {
                    let output_gate_signal = infos_context_to_context(cesure, &signal);

                    if let Some(ref mut signal) = memory_gate_signal {
                        signal.add(&output_gate_signal);
                    } else {
                        memory_gate_signal = Some(output_gate_signal);
                    }

                    let (signal, weights_changes) = cesure.memory_gate.backpropagate_no_change(&outputs[i-1].memory_out, &memory_gate_signal.unwrap(), learning_rate,
                                                                         &memory_gate_last_changes, momentum);
                    weights_changes_add_or_clone(&mut memory_gate_changes_sum, &weights_changes);
                    memory_gate_nb_changes += 1.0;
                    memory_gate_last_changes = Some(weights_changes);

                    let signal = infos_context_output_to_context(cesure, &signal);
                    memory_gate_signal = Some(signal);

                }
            }
        }

        weights_changes_div(&mut output_gate_changes_sum, output_gate_nb_changes);
        weights_changes_div(&mut memory_gate_changes_sum, memory_gate_nb_changes);

        cesure.output_gate.apply_changes(&output_gate_changes_sum);
        cesure.memory_gate.apply_changes(&memory_gate_changes_sum);

        if show {
            println!("Epoch #{}, Error = {}, LearningRate = {}, Momentum = {}", iteration, error_sum, learning_rate, momentum);
        }
    }

    stdin.read_line();
    println!("Training finished!");
    println!("Type the file name to save Cesure in (type nothing if you don't want to save) :");
    let answer = stdin.read_line_blocking();
    match answer.as_str() {
        "" => {},
        _ => {
            cesure.save(answer.as_str());
            println!("Saved in {}", answer.as_str());
        },
    }
}

pub fn train_mod2(cesure: &mut Cesure, training_sets: &Vec<TrainingSet>, error_calculation: &ErrorCalculation, learning_rate: f64, momentum: f64, depth: usize, iterations: usize) {

    let mut stdin = AsyncStdinReader::new();
    let mut learning_rate = learning_rate;
    let mut momentum = momentum;
    let mut iterations = iterations;
    let mut show = true;

    let mut output_gate_last_changes = None;
    let mut memory_gate_last_changes = None;


    for iteration in 0..iterations {

        if let Some(line) = stdin.read_line() {
            let mut args = line.as_str().split_whitespace();
            match args.next() {
                Some(arg) => {
                    match arg {
                        "stop" => { break; },
                        "show" => { show = true; },
                        "hide" => { show = false; },
                        "setlr" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(lr) => {
                                            match lr < 0.0 {
                                                true => { println!("Learning rate has to be positive"); },
                                                false => { learning_rate = lr; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setlr"); }
                            }
                        }
                        "setmom" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mom) => {
                                            match mom < 0.0 {
                                                true => { println!("Momentum has to be positive"); },
                                                false => { momentum = mom; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmom"); }
                            }
                        }
                        /*"setiters" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(iters) => {
                                            iterations = iters;
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setiters"); }
                            }
                        }*/
                        _ => println!("Unknown command [{}]", line),
                    }
                },
                None => {}
            }
        }

        let mut error_sum = 0.0;

        for set_i in 0..training_sets.len() {
            let training_set = &training_sets[set_i];
            let sequence_len = training_set.compute_sequence.len();

            cesure.new_sequence(&training_set.infos);
            for injection in &training_set.inject_sequence {
                cesure.inject_next(injection);
            }
            let mut memory_outputs = Vec::with_capacity(sequence_len);

            for tick_i in 0..sequence_len {

                let output = cesure.compute_next_verbose();
                let output_output = output.output_out;
                memory_outputs.push(cesure.compute_next_verbose().memory_out);

                let error = (error_calculation.calculate)(&output_output.output, &training_set.compute_sequence[tick_i]);
                error_sum += error.get_abs_avg();

                let (output_gate_signal, changes) = cesure.output_gate.backpropagate(&output_output, &error, learning_rate, output_gate_last_changes, momentum);
                output_gate_last_changes = Some(changes);

                let mut memory_gate_signal = infos_context_to_context(cesure, &output_gate_signal);
                let mut memory_gate_changes = Vec::new();

                for back_i in 0..tick_i {
                    if back_i < depth {
                        let back_i = tick_i - 1 - back_i;

                        let (signal, changes) = cesure.memory_gate.backpropagate_no_change(&memory_outputs[back_i], &memory_gate_signal, learning_rate, &memory_gate_last_changes, momentum);
                        memory_gate_signal = infos_context_output_to_context(cesure, &signal);
                        weights_changes_add_or_clone(&mut memory_gate_changes, &changes);
                        memory_gate_last_changes = Some(changes);
                    }
                }

                if memory_gate_changes.len() > 0 {
                    cesure.memory_gate.apply_changes(&memory_gate_changes);
                }
            }
        }

        if show {
            println!("Epoch #{}, Error = {}, LearningRate = {}, Momentum = {}", iteration, error_sum, learning_rate, momentum);
        }

    }

    stdin.read_line();
    println!("Training finished!");
    println!("Type the file name to save Cesure in (type nothing if you don't want to save) :");
    let answer = stdin.read_line_blocking();
    match answer.as_str() {
        "" => {},
        _ => {
            cesure.save(answer.as_str());
            println!("Saved in {}", answer.as_str());
        },
    }
}

fn infos_context_to_context(cesure : &Cesure, infos_context : &Matrix) -> Matrix {
    assert!(infos_context.is_row() && infos_context.len == INFOS_DIMENSION+cesure.context_dimension);
    let begin = INFOS_DIMENSION;
    let end = INFOS_DIMENSION+cesure.context_dimension;
    let mut vec = Vec::with_capacity(cesure.context_dimension);
    for i in begin..end {
        vec.push(infos_context.datas[i]);
    }
    assert!(vec.len() == cesure.context_dimension);
    return Matrix::new_row_from_datas(vec);
}
fn infos_context_output_to_context(cesure : &Cesure, infos_context_output : &Matrix) -> Matrix {
    assert!(infos_context_output.is_row() && infos_context_output.len == INFOS_DIMENSION+cesure.context_dimension+CHORD_DIMENSION);
    let begin = INFOS_DIMENSION;
    let end = INFOS_DIMENSION+cesure.context_dimension;
    let mut vec = Vec::with_capacity(cesure.context_dimension);
    for i in begin..end {
        vec.push(infos_context_output.datas[i]);
    }
    assert!(vec.len() == cesure.context_dimension);
    return Matrix::new_row_from_datas(vec);
}

fn weights_changes_div(weights_changes : &mut Vec<Matrix>, val : f64) {
    for i in 0..weights_changes.len() {
        weights_changes[i].div_scl(val);
    }
}
fn weights_changes_add_or_clone(weights_changes: &mut Vec<Matrix>, to_add: &Vec<Matrix>) {
    if weights_changes.len() == 0 {
        weights_changes.clone_from(to_add);
    } else {
        assert!(weights_changes.len() == to_add.len());
        for i in 0..weights_changes.len() {
            weights_changes[i].add(&to_add[i]);
        }
    }
}



/*pub fn train(cesure : &mut Cesure, training_set : &TrainingSet, error_calculation: &ErrorCalculation, iterations : usize, learning_rate : f64, momentum : f64) {
    let sequence_len = training_set.compute_sequence.len();

    for iteration in 0..iterations {
        cesure.new_sequence(&training_set.infos);

        for injection in &training_set.inject_sequence {
            cesure.inject_next(injection);
        }

        let mut outputs = Vec::with_capacity(sequence_len);
        let mut errors = Vec::with_capacity(sequence_len);

        for i in 0..sequence_len {
            outputs.push( cesure.compute_next_verbose() );
            errors.push( (error_calculation.calculate)(&outputs[i].output_out.output, &training_set.compute_sequence[i]) );
        }


        let mut output_gate_changes_sum = Vec::new();
        let mut output_gate_nb_changes = 0.0;
        let mut memory_gate_changes_sum = Vec::new();
        let mut memory_gate_nb_changes = 0.0;

        let mut output_gate_last_changes = None;
        let mut memory_gate_last_changes = None;
        let mut memory_gate_signal : Option<Matrix> = None;

        for i in 0..sequence_len {
            let i = sequence_len-1 - i;

            let tmp = cesure.output_gate.backpropagate_no_change(&outputs[i].output_out, &errors[i], learning_rate,
                                                                                    &output_gate_last_changes, momentum);
            weights_changes_add_or_clone(&mut output_gate_changes_sum, &tmp.1);
            output_gate_nb_changes += 1.0;
            output_gate_last_changes = Some(tmp.1);

            if i > 1 {
                let output_gate_signal = infos_context_to_context(cesure, &tmp.0);

                if memory_gate_signal.is_some() {
                    let mut new_signal = memory_gate_signal.unwrap();
                    new_signal.add(&output_gate_signal);
                    memory_gate_signal = Some(new_signal);
                } else {
                    memory_gate_signal = Some(output_gate_signal);
                }

                let tmp = cesure.memory_gate.backpropagate_no_change(&outputs[i-1].memory_out, &memory_gate_signal.unwrap(), learning_rate,
                                                                                      &memory_gate_last_changes, momentum);
                weights_changes_add_or_clone(&mut memory_gate_changes_sum, &tmp.1);
                memory_gate_nb_changes += 1.0;
                memory_gate_last_changes = Some(tmp.1);

                let signal = infos_context_output_to_context(cesure, &tmp.0);
                memory_gate_signal = Some(signal);

            }

        }

        weights_changes_div(&mut output_gate_changes_sum, output_gate_nb_changes);
        weights_changes_div(&mut memory_gate_changes_sum, memory_gate_nb_changes);

        cesure.output_gate.apply_changes(&output_gate_changes_sum);
        cesure.memory_gate.apply_changes(&memory_gate_changes_sum);

        let mut error_sum = 0.0;
        for error in errors {
            error_sum += error.get_abs_avg();
        }
        //error_sum = cesure.calculate_error_sum(&training_set);
        println!("Epoch #{}, Error = {}", iteration, error_sum);
    }

}*/
/*pub fn train (cesure : &mut Cesure, training_set : &TrainingSet, iterations : usize, backpropagation_depth : usize,  learning_rate : f64, momentum : f64) {
    let sequence_len = training_set.compute_sequence.len();

    for iteration in 0..iterations {
        cesure.new_sequence(&training_set.infos);

        for injection in &training_set.inject_sequence {
            cesure.inject_next(injection);
        }

        let mut outputs = Vec::with_capacity(sequence_len);
        let mut error_sum = 0.0;

        let mut output_gate_last_changes = None;
        let mut memory_gate_last_changes = None;

        for i in 0..sequence_len {
            outputs.push( cesure.compute_next_verbose() );
            let error = sub(&outputs[i].output_out.output, &training_set.compute_sequence[i]);
            error_sum += error.get_abs_avg();

            let tmp = cesure.output_gate.backpropagate(&outputs[i].output_out, &error, learning_rate,
                                                       output_gate_last_changes, momentum);
            let mut signal = tmp.0;
            output_gate_last_changes = Some(tmp.1);
            if i > 1 {
                signal = infos_context_to_context(cesure, &signal);
                let mut last_i = i - 1;
                loop {
                    let tmp = cesure.memory_gate.backpropagate(&outputs[last_i].memory_out, &signal, learning_rate,
                                                               memory_gate_last_changes, momentum);
                    signal = tmp.0;
                    memory_gate_last_changes = Some(tmp.1);
                    signal = infos_context_output_to_context(cesure, &signal);
                    if last_i > i-backpropagation_depth && last_i > 0 {
                        last_i -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
        println!("Epoch #{}, Error = {}", iteration, error_sum);
    }

}*/
/*pub struct Backpropagation<'t> {
    pub cesure : &'t mut Cesure,
    pub last_output_gate_changes : Vec<Matrix>,
    pub last_memory_gate_changes : Vec<Matrix>,
}


#[allow(dead_code)]
impl<'t> Backpropagation<'t> {

    pub fn new(cesure : &'t mut Cesure) -> Backpropagation<'t> {
        let mut last_output_gate_changes = Vec::with_capacity(cesure.output_gate.nb_layers);
        let mut last_memory_gate_changes = Vec::with_capacity(cesure.memory_gate.nb_layers);
        for i in 0..cesure.output_gate.nb_layers {
            last_output_gate_changes.push( Matrix::new(cesure.output_gate.layers[i].rows, cesure.output_gate.layers[i].cols) );
        }
        for i in 0..cesure.memory_gate.nb_layers {
            last_memory_gate_changes.push( Matrix::new(cesure.memory_gate.layers[i].rows, cesure.memory_gate.layers[i].cols) );
        }
        return Backpropagation {
            cesure : cesure,
            last_output_gate_changes : last_output_gate_changes,
            last_memory_gate_changes : last_memory_gate_changes,
        }
    }

    /*pub fn train_this (&mut self, training_set : &TrainingSet, iterations : usize, learning_rate : f64, momentum : f64) {
        let sequence_len = training_set.compute_sequence.len();
        let mut error_sum = 0.0;

        for iteration in 0..iterations {
            self.cesure.new_sequence(&training_set.infos);

            for i in 0..training_set.inject_sequence.len() {
                self.cesure.inject_next(&training_set.inject_sequence[i]);
            }

            let mut outputs = Vec::with_capacity(sequence_len);
            error_sum = 0.0;

            for i in 0..sequence_len {
                outputs.push( self.cesure.compute_next_verbose() );
                let error = sub(&outputs[i].output_out.output, &training_set.compute_sequence[i]);
                error_sum += error.clone().abs().get_avg();
                let mut signal = self.cesure.output_gate.backpropagate_error_signal(&outputs[i].output_out, &error, learning_rate);
                if i > 1 {
                    signal = Backpropagation::infos_context_to_context(&signal);
                    let mut last_i = i - 1;
                    loop {
                        signal = self.cesure.memory_gate.backpropagate_error_signal(&outputs[last_i].memory_out, &signal, learning_rate);
                        signal = Backpropagation::infos_context_output_to_context(&signal);
                        if last_i > 0 {
                            last_i -= 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            println!("Epoch #{}, Error = {}", iteration, error_sum);
        }

    }*/



}



/*#[allow(dead_code)]
pub fn iteration (cesure : &mut Cesure, training_set : &TrainingSet, learning_rate : f64) -> f64 {
    cesure.new_sequence(&training_set.infos);

    for i in 0..training_set.inject_sequence.len() {
        cesure.inject_next(&training_set.inject_sequence[i]);
    }

    let mut error_sum = 0.0;
    let sequence_len = training_set.compute_sequence.len();
    let mut outputs = Vec::with_capacity(sequence_len);
    for i in 0..sequence_len {
        outputs.push( cesure.compute_next_verbose() );
        let error = sub(&outputs[i].output_out.output, &training_set.compute_sequence[i]);
        error_sum += error.clone().abs().get_avg();
        let signal = cesure.output_gate.backpropagate_error_signal(&outputs[i].output_out, &error, learning_rate);
        if i > 1 {
            let mut context_signal = row_slice(&signal, cesure::INFOS_DIMENSION).1;
            let mut last_i = i - 1;
            loop {
                context_signal = cesure.memory_gate.backpropagate_error_signal(&outputs[last_i].memory_out, &context_signal, learning_rate);
                context_signal = row_slice(&context_signal, cesure::INFOS_DIMENSION).1;
                context_signal = row_slice(&context_signal, cesure::CONTEXT_DIMENSION).0;

                if last_i > 0 {
                    last_i -= 1;
                } else {
                    break;
                }
            }
        }
    }

    error_sum
}*/*/