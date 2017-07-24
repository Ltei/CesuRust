
extern crate rand;
extern crate crossbeam;

use std::f64;
use std::sync::{Arc, Mutex};

use network::cesure::Cesure;
use network::training::error_calculation::ErrorCalculation;
use network::training::training_set::TrainingSet;

use utils::traits::Parse;
use utils::io::{AsyncStdinReader, AsyncStdinRead};
use utils::io::stdin_readline;



struct CesureAndErrors {
    pub cesure : Cesure,
    pub errors : Vec<f64>,
}

struct CesureAndError {
    pub cesure: Cesure,
    pub error: f64,
}


/**
* Train a Cesure object using a simple Genetic algorithm
* @input cesure : The cesure object to train
* @input magnitude_0 : The amount of changes at iteration0
* @input magnitude_1 : The amount of changes at iteration1
* @input iterations : The number of iterations
* @input training_sets : The TrainingSet objects to calculate the error on
*/
pub fn train(cesure : &mut Cesure, training_sets: &Vec<TrainingSet>, error_calculation: &ErrorCalculation, magnitude_0: f64, magnitude_1: f64, iterations: usize) {

    let mut best_cesure = {
        let mut first_cesure = cesure.clone();
        let first_error = first_cesure.calculate_error_sum_multi(&training_sets, &error_calculation);
        CesureAndError {cesure: first_cesure, error: first_error}
    };

    let nb_threads = 4;
    let mut stdin = AsyncStdinReader::new();
    let mut show = true;
    let mut iterations = iterations;
    let mut magnitude_0 = magnitude_0;
    let mut magnitude_1 = magnitude_1;

    for iteration in 0..iterations {

        if let Some(line) = stdin.read_line() {
            let mut args = line.as_str().split_whitespace();
            match args.next() {
                Some(arg) => {
                    match arg {
                        "stop" => { break; },
                        "show" => { show = true; },
                        "hide" => { show = false; },
                        "setmag0" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mag) => {
                                            match mag < 0.0 {
                                                true => { println!("Magnitude has to be positive"); },
                                                false => { magnitude_0 = mag; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmag0"); }
                            }
                        }
                        "setmag1" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mag) => {
                                            match mag < 0.0 {
                                                true => { println!("Magnitude has to be positive"); },
                                                false => { magnitude_1 = mag; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmag0"); }
                            }
                        }
                        "setiters" => {
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
                        }
                        _ => println!("Unknown command [{}]", line),
                    }
                },
                None => {}
            }
        }

        let actual_magnitude = {
            let magnitude_x = (iteration as f64) / (iterations as f64);
            magnitude_x * magnitude_1 + (1.0 - magnitude_x) * magnitude_0
        };
        if actual_magnitude.is_infinite() || actual_magnitude.is_nan() || actual_magnitude <= 0.0 {
            println!("Invalid magnitude ({}), breaking train", actual_magnitude);
            break;
        }

        let mut threads = Vec::with_capacity(nb_threads);
        crossbeam::scope(|scope| {
            for _ in 0..nb_threads {
                let thread = scope.spawn(|| iteration_mod1(&best_cesure, &training_sets, &error_calculation, actual_magnitude));
                threads.push(thread);
            }
        });

        for thread in threads {
            if let Some(cesure_and_error) = thread.join() {
                if cesure_and_error.error < best_cesure.error {
                    best_cesure = cesure_and_error;
                }
            }
        }

        if show {
            print!("{:<20}", format!("Epoch #{}, Error = {}", iteration, best_cesure.error));
            println!(", Magnitude = {} ({}, {})", actual_magnitude, magnitude_0, magnitude_1);
        }

    }

    cesure.clone_from(&best_cesure.cesure);

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
fn iteration_mod1(cesure: &CesureAndError, training_sets: &Vec<TrainingSet>, error_calculation: &ErrorCalculation, magnitude: f64) -> Option<CesureAndError> {
    let mut computed_cesure = cesure.cesure.clone_randomized(magnitude, &mut rand::thread_rng());
    let computed_error = computed_cesure.calculate_error_sum_multi(&training_sets, &error_calculation);
    if computed_error > cesure.error {
        return None;
    } else {
        return Some(CesureAndError {cesure: computed_cesure, error: computed_error});
    }
}

pub fn train_mod2(cesure : &mut Cesure, training_sets: &Vec<TrainingSet>, error_calculation: &ErrorCalculation, magnitude_0: f64, magnitude_1: f64, iterations: usize) {

    let mut best_cesure = cesure.clone();

    let mut stdin = AsyncStdinReader::new();
    let mut show = true;
    let mut iterations = iterations;
    let mut magnitude_0 = magnitude_0;
    let mut magnitude_1 = magnitude_1;

    let mut error_sum = 0.0;

    let mut training_set_i = 0;
    let mut training_set_tick_i = 0;

    for iteration in 0..iterations {

        if let Some(line) = stdin.read_line() {
            let mut args = line.as_str().split_whitespace();
            match args.next() {
                Some(arg) => {
                    match arg {
                        "stop" => { break; },
                        "show" => { show = true; },
                        "hide" => { show = false; },
                        "setmag0" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mag) => {
                                            match mag < 0.0 {
                                                true => { println!("Magnitude has to be positive"); },
                                                false => { magnitude_0 = mag; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmag0"); }
                            }
                        }
                        "setmag1" => {
                            match args.next() {
                                Some(arg) => {
                                    match arg.parse() {
                                        Ok(mag) => {
                                            match mag < 0.0 {
                                                true => { println!("Magnitude has to be positive"); },
                                                false => { magnitude_1 = mag; },
                                            }
                                        }
                                        Err(msg) => { println!("{}", msg); }
                                    }
                                },
                                None => { println!("No argument on command setmag0"); }
                            }
                        }
                        "setiters" => {
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
                        }
                        _ => println!("Unknown command [{}]", line),
                    }
                },
                None => {}
            }
        }

        let actual_magnitude = {
            let magnitude_x = (iteration as f64) / (iterations as f64);
            magnitude_x * magnitude_1 + (1.0 - magnitude_x) * magnitude_0
        };
        if actual_magnitude.is_infinite() || actual_magnitude.is_nan() || actual_magnitude <= 0.0 {
            println!("Invalid magnitude ({}), breaking train", actual_magnitude);
            break;
        }


        let mut new_cesure = best_cesure.clone_randomized(actual_magnitude, &mut rand::thread_rng());

        let mut best_cesure_error = (error_calculation.calculate)(&best_cesure.compute_next(), &training_sets[training_set_i].compute_sequence[training_set_tick_i]).get_abs_avg();
        let new_cesure_error = (error_calculation.calculate)(&new_cesure.compute_next(), &training_sets[training_set_i].compute_sequence[training_set_tick_i]).get_abs_avg();
        if new_cesure_error < best_cesure_error {
            best_cesure = new_cesure;
            best_cesure_error = new_cesure_error;
        }

        error_sum += best_cesure_error;


        if show {
            if iteration % 100 == 0 {
                print!("{:<20}", format!("Epoch #{}, Error = {}", iteration, best_cesure_error));
                print!(", Magnitude = {} ({}, {}), ", actual_magnitude, magnitude_0, magnitude_1);
                println!("Training_set = {}, Training_set_tick : {}", training_set_i, training_set_tick_i);
            }
        }

        training_set_tick_i += 1;
        if training_set_tick_i >= training_sets[training_set_i].compute_sequence.len() {
            println!("ERROR_SUM : {}", error_sum);
            error_sum = 0.0;
            training_set_i += 1;
            training_set_tick_i = 0;
            if training_set_i >= training_sets.len() {
                training_set_i = 0;
            }
        }

    }

    cesure.clone_from(&best_cesure);


    println!("Training finished!");
    ui_save_cesure(&cesure, &mut stdin);
}


fn ui_save_cesure(cesure: &Cesure, stdin: &mut Arc<Mutex<AsyncStdinReader>>) {
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


/*fn ui_save_cesure(cesure: &Cesure) {
    println!("Type the file name to save Cesure in (type nothing if you don't want to save) :");
    let answer = stdin_readline();
    match answer.as_str() {
        "" => {},
        _ => {
            cesure.save(answer.as_str());
            println!("Saved in {}", answer.as_str());
        },
     }
}*/
/*fn train_command_handler(channel: DoubleI32Channel) -> thread::JoinHandle<()> {
    thread::spawn(|| {
        loop {
            let buffer = stdin_readline();
            match buffer.as_str() {
                "show" => { println!(":::Command show"); },
                "hide" => { println!(":::Command hide"); },
                "stop" => println!(":::Command stop"),
                _ => println!(":::Unknown command line : ---{}---", buffer),
            }
        }
    })
}*/
/*/**
* Return an Option containing :
* - Some<(Cesure,f64)> : If the new Cesure object is better,
*                        returns a tuple containing it and it's error
* - None : If the paramater Cesure object is better
* @input cesure : The cesure object to train
* @input cesure_error : The cesure object's error on the TrainingSet
* @input training_sets : The TrainingSet objects to calculate the error on
* @input magnitude : The magnitude for changes
* @input rand : A ThreadRng object
*/
fn genetic_iteration_multi( cesure_and_errors: &CesureAndErrors, training_sets: &Vec<TrainingSet>, error_calculations: &Vec<ErrorCalculation>,
                            magnitude: f64, rand: &mut rand::ThreadRng ) -> CesureAndErrors {
    let mut new_cesure = cesure_and_errors.cesure.clone_randomized(magnitude, rand);
    let mut new_error = Vec::with_capacity();
    for i in 0..error_calculations.len() {
        let error = new_cesure.calculate_error_sum_multi(training_sets, &error_calculations[i]);
        new_error += error;
    }
    CesureAndError {cesure: new_cesure, error: new_error}
}*/
/*/**
* Train a Cesure object using a simple Genetic algorithm
* @input cesure : The cesure object to train
* @input magnitude_0 : The amount of changes at iteration0
* @input magnitude_1 : The amount of changes at iteration1
* @input iterations : The number of iterations
* @input training_set : The TrainingSet object to calculate the error on
*/
pub fn train(cesure : &mut Cesure, training_set : &TrainingSet, error_calculation: &ErrorCalculation,
             magnitude_0 : f64, magnitude_1 : f64, iterations : usize) {
    let mut rand = rand::thread_rng();

    let mut best_cesure = cesure.clone();
    let best_error = best_cesure.calculate_error_sum(&training_set, error_calculation);

    let mut best = CesureAndError::new(best_cesure, best_error);

    for i in 0..iterations {
        let magnitude_x = (i as f64) / (iterations as f64);
        let actual_magnitude = magnitude_x * magnitude_1 + (1.0-magnitude_x) * magnitude_0;
        match genetic_iteration(&best, training_set, error_calculation, actual_magnitude, &mut rand) {
            Some(x) => best = x,
            None => {},
        }
        println!("Epoch #{}, Error = {}", i, best.error);
    }
    cesure.clone_from(&best.cesure);
}

/**
* Return an Option containing :
* - Some<(Cesure,f64)> : If the new Cesure object is better,
*                        returns a tuple containing it and it's error
* - None : If the paramater Cesure object is better
* @input cesure : The cesure object to train
* @input cesure_error : The cesure object's error on the TrainingSet
* @input training_set : The TrainingSet object to calculate the error on
* @input magnitude : The magnitude for changes
* @input rand : A ThreadRng object
*/
fn genetic_iteration( cesure_and_error : &CesureAndError, training_set : &TrainingSet, error_calculation: &ErrorCalculation,
                      magnitude : f64, rand : &mut rand::ThreadRng ) -> Option<(CesureAndError)> {
    let mut new_cesure = cesure_and_error.cesure.clone_randomized(magnitude, rand);
    let new_error = new_cesure.calculate_error_sum(&training_set, error_calculation);
    if new_error < cesure_and_error.error {
        return Some(CesureAndError::new(new_cesure,new_error));
    } else {
        return None;
    }
}*/
/*pub fn train(cesure : &mut Cesure, magnitude_max : f64, magnitude_min : f64, iterations : usize, training_set : &TrainingSet) {
    let mut rand = rand::thread_rng();

    let mut best_cesure = cesure.clone();
    let mut best_error = best_cesure.calculate_error_sum(&training_set);

    for i in 0..iterations {
        let magnitude_x = (i as f64) / (iterations as f64);
        let actual_magnitude = magnitude_x * magnitude_min + (1.0-magnitude_x) * magnitude_max;
        let mut new_cesure = best_cesure.clone_randomized(actual_magnitude, &mut rand);
        let new_error = new_cesure.calculate_error_sum(&training_set);
        if new_error < best_error {
            best_cesure = new_cesure;
            best_error = new_error;
        }
        println!("Epoch #{} - Error = {}", i, best_error);
    }
    cesure.clone_from(&best_cesure);
}*/