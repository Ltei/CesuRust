#![allow(dead_code)]

extern crate rand;

use rand::Rng;

use networkV2::cesure::Cesure;
use networkV2::training::training_set::TrainingSet;
use networkV2::training::cesure_and_error::CesureAndError;




/**
* Train a Cesure object using a simple Genetic algorithm
* @input cesure : The cesure object to train
*
* @input iterations : The number of iterations
* @input training_set : The TrainingSet object to calculate the error on
*/
pub fn train(cesure : &mut Cesure, magnitude_0 : f64, magnitude_1 : f64, iterations : usize, training_set : &TrainingSet) {
    let mut rand = rand::thread_rng();

    let mut best_cesure = cesure.clone();
    let best_error = best_cesure.calculate_error_sum(&training_set);

    let mut best = CesureAndError::new(best_cesure, best_error);

    for i in 0..iterations {
        let magnitude_x = (i as f64) / (iterations as f64);
        let actual_magnitude = magnitude_x * magnitude_1 + (1.0-magnitude_x) * magnitude_0;
        match genetic_iteration(&best, training_set, actual_magnitude, &mut rand) {
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
#[inline]
fn genetic_iteration( cesure_and_error : &CesureAndError, training_set : &TrainingSet, magnitude : f64, rand : &mut rand::ThreadRng ) -> Option<(CesureAndError)> {
    let mut new_cesure = cesure_and_error.cesure.clone_randomized(magnitude, rand);
    let new_error = new_cesure.calculate_error_sum(&training_set);
    if new_error < cesure_and_error.error {
        return Some(CesureAndError::new(new_cesure,new_error));
    } else {
        return None;
    }
}


pub fn train_multi(cesure : &mut Cesure, magnitude_0 : f64, magnitude_1 : f64, iterations : usize, training_sets : &Vec<TrainingSet>) {
    let mut rand = rand::thread_rng();

    let mut best_cesure = cesure.clone();
    let best_error = best_cesure.calculate_error_sum_multi(training_sets);

    let mut best = CesureAndError::new(best_cesure, best_error);

    for i in 0..iterations {
        let magnitude_x = (i as f64) / (iterations as f64);
        let actual_magnitude = magnitude_x * magnitude_1 + (1.0-magnitude_x) * magnitude_0;
        match genetic_iteration_multi(&best, training_sets, actual_magnitude, &mut rand) {
            Some(x) => best = x,
            None => {},
        }
        println!("Epoch #{}, Error = {}", i, best.error);
    }
    cesure.clone_from(&best.cesure);
}

#[inline]
fn genetic_iteration_multi( cesure_and_error : &CesureAndError, training_sets : &Vec<TrainingSet>, magnitude : f64, rand : &mut rand::ThreadRng ) -> Option<(CesureAndError)> {
    let mut new_cesure = cesure_and_error.cesure.clone_randomized(magnitude, rand);
    let new_error = new_cesure.calculate_error_sum_multi(training_sets);
    if new_error < cesure_and_error.error {
        return Some(CesureAndError::new(new_cesure,new_error));
    } else {
        return None;
    }
}