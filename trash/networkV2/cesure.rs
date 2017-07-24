#![allow(dead_code)]

extern crate rand;

use std::clone::Clone;
use std::ops::Add;
use std::fs;
use std::path::Path;

use networkV2::activation::Activation;
use networkV2::activation::ActivationType;
use networkV2::feedforward_gate;
use networkV2::feedforward_gate::FeedforwardGate;
use networkV2::training::training_set::TrainingSet;
use networkV2::training::genetic;

use networkV2::music::CesureMusic;
use networkV2::music::DIVISION_RANGE;
use networkV2::music::NB_TICKS_RANGE;
use networkV2::music::INFOS_DIMENSION;
use networkV2::music::CHORD_DIMENSION;

use utils::matrix::Matrix;
use utils::matrix_math::sub;
use utils::matrix_math::row_concatenate;
use utils::traits::Parse;



pub struct Cesure {
    pub infos_dimension : usize,
    pub context_dimension : usize,
    pub output_dimension : usize,
    pub note_gates : Vec<FeedforwardGate>,
    pub memory_gate : FeedforwardGate,
    pub infos : Matrix,
    pub context : Matrix,
}

impl Clone for Cesure {
    fn clone(&self) -> Cesure {
        return Cesure {
            infos_dimension : self.infos_dimension,
            context_dimension : self.context_dimension,
            output_dimension : self.output_dimension,
            note_gates : self.note_gates.clone(),
            memory_gate : self.memory_gate.clone(),
            infos : self.infos.clone(),
            context : self.context.clone(),
        }
    }
    fn clone_from(&mut self, source: &Cesure) {
        self.infos_dimension = source.infos_dimension;
        self.context_dimension = source.context_dimension;
        self.output_dimension = source.output_dimension;
        self.note_gates.clone_from(&source.note_gates);
        self.memory_gate.clone_from(&source.memory_gate);
        self.infos.clone_from(&source.infos);
        self.context.clone_from(&source.context);
    }
}
impl Parse for Cesure {
    fn to_string(&self) -> String {
        let mut output = format!("{} {} {}", self.infos_dimension, self.context_dimension, self.output_dimension);
        for i in 0..self.note_gates.len() {
            output = output.add(format!("\nNOTE_GATE_{}\n", i).as_str());
            output = output.add(self.note_gates[i].to_string().as_str());
        }
        output = output.add("\nMEMORY_GATE\n");
        output = output.add(self.memory_gate.to_string().as_str());
        return output;
    }
    fn from_string(str : &str) -> Cesure {
        let lines : Vec<&str> = str.split("\n").collect();

        let header : Vec<&str> = lines[0].split(" ").collect();
        assert!(header.len() == 3);
        let infos_dimension = header[0].parse().unwrap();
        let context_dimension = header[1].parse().unwrap();
        let output_dimension = header[2].parse().unwrap();

        let mut note_gates = Vec::with_capacity(CHORD_DIMENSION);
        for i in 0..(CHORD_DIMENSION-1) {
            let gate_str : Vec<&str> = str.split(format!("\nNOTE_GATE_{}\n", i).as_str()).collect();
            let gate_str : Vec<&str> = gate_str[1].split(format!("\nNOTE_GATE_{}\n", i+1).as_str()).collect();
            let gate_str = gate_str[0];
            note_gates.push(FeedforwardGate::from_string(gate_str));
        }
        let gate_str : Vec<&str> = str.split(format!("\nNOTE_GATE_{}\n", CHORD_DIMENSION-1).as_str()).collect();
        let gate_str : Vec<&str> = gate_str[1].split(format!("\nMEMORY_GATE\n").as_str()).collect();
        note_gates.push(FeedforwardGate::from_string(gate_str[0]));

        let memory_gate = FeedforwardGate::from_string(gate_str[1]);

        return Cesure {
            infos_dimension : infos_dimension,
            context_dimension : context_dimension,
            output_dimension : output_dimension,
            note_gates : note_gates,
            memory_gate : memory_gate,
            infos : Matrix::new_row(infos_dimension),
            context : Matrix::new_row(context_dimension),
        }
    }
}

impl Cesure {

    /**
    * Constructor
    */
    pub fn new(context_dimension: usize) -> Cesure {
        let infos_context_dimension = INFOS_DIMENSION + context_dimension;
        let infos_context_output_dimension = infos_context_dimension + CHORD_DIMENSION;

        let mut note_gates = Vec::with_capacity(CHORD_DIMENSION);
        for i in 0..CHORD_DIMENSION {
            note_gates.push(FeedforwardGate::new_auto(infos_context_dimension+i, 1, 4, Activation::new(&ActivationType::Sigmoid)));
        }

        return Cesure {
            infos_dimension: INFOS_DIMENSION,
            context_dimension: context_dimension,
            output_dimension: CHORD_DIMENSION,
            note_gates: note_gates,
            memory_gate: FeedforwardGate::new_auto(infos_context_output_dimension, context_dimension, 4, Activation::new(&ActivationType::Sigmoid)),
            infos: Matrix::new_row(INFOS_DIMENSION),
            context: Matrix::new_row(context_dimension),
        }
    }

    /**
    * Return a cloned object with some random changes
    * @input magnitude : The amount of random to apply
    * @input rand : The ThreadRng object to use for random
    * @return The cloned Cesure object
    */
    pub fn clone_randomized(&self, magnitude : f64, rand : &mut rand::ThreadRng) -> Cesure {
        let mut note_gates = Vec::with_capacity(CHORD_DIMENSION);
        for i in 0..CHORD_DIMENSION {
            note_gates.push(self.note_gates[i].clone_randomized(magnitude, rand));
        }
        return Cesure {
            infos_dimension : self.infos_dimension,
            context_dimension : self.context_dimension,
            output_dimension : self.output_dimension,
            note_gates : note_gates,
            memory_gate : self.memory_gate.clone_randomized(magnitude, rand),
            infos : self.infos.clone(),
            context : self.context.clone(),
        }
    }

    /**
    * Print all cesure's layers
    * @input s : A title printed at the beginning
    */
    pub fn print(&self, s :& str) {
        println!("{}",s);
        self.infos.print_title("- Infos :");
        self.context.print_title("- Context :");
        for i in 0..CHORD_DIMENSION {
            println!("- Note Gate {} :", i);
            self.note_gates[i].print();
        }
        println!("- Memory Gate :");
        self.memory_gate.print();
    }

    pub fn get_nb_neurons(&self) -> usize {
        let mut nb_neurons = self.memory_gate.get_nb_neurons();
        for gate in &self.note_gates {
            nb_neurons += gate.get_nb_neurons();
        }
        nb_neurons
    }
    pub fn print_nb_neurons(&self) {
        println!("Memory gate nb neurons : {}", self.memory_gate.get_nb_neurons());
        for i in 0..self.note_gates.len() {
            println!("Note gate {} nb neurons : {}", i, self.note_gates[i].get_nb_neurons());
        }
        println!("Total nb neurons : {}", self.get_nb_neurons());
    }

    /**
    * Prepare Cesure for a new output sequence
    * @input infos : The infos to use for this new sequence
    */
    pub fn new_sequence(&mut self, infos : &Matrix) {
        assert!(infos.is_row() && infos.len == INFOS_DIMENSION);
        self.infos.clone_from(infos);
        self.context.set_zero();
    }

    /**
    * Compute the next output of the current sequence
    * @return The computed output
    */
    pub fn compute_next(&mut self) -> Matrix {
        let mut context = row_concatenate(&self.infos, &self.context);

        let mut output = self.note_gates[0].compute(&context);
        context.row_append(output.datas[0]);
        for i in 1..CHORD_DIMENSION {
            let out = self.note_gates[i].compute(&context).datas[0];
            output.row_append(out);
            context.row_append(out);
        }

        let memory_out = self.memory_gate.compute(&context);
        self.context = memory_out;

        output
    }

    /**
    * Manually inject the next output, changing the context
    * @input input : The input to inject
    */
    pub fn inject_next(&mut self, input : &Matrix) {
        let mut context = row_concatenate(&self.infos, &self.context);
        context.row_concatenate(input);;
        let memory_out = self.memory_gate.compute(&context);
        self.context = memory_out;
    }

    /**
    * Return the error of this network, on a TrainingSet
    * @input training_set : The TrainingSet object to calculate the error on
    * @return The error
    */
    pub fn calculate_error_sum(&mut self, training_set : &TrainingSet) -> f64 {
        self.new_sequence(&training_set.infos);
        for i in 0..training_set.inject_sequence.len() {
            self.inject_next(&training_set.inject_sequence[i]);
        }
        let mut error_sum = 0.0;
        for i in 0..training_set.compute_sequence.len() {
            error_sum += self.compute_next().sub(&&training_set.compute_sequence[i]).abs().get_avg();
        }
        error_sum
    }
    pub fn calculate_error_sum_multi(&mut self, training_sets : &Vec<TrainingSet>) -> f64 {
        let mut error_sum = 0.0;
        for training_set in training_sets {
            error_sum += self.calculate_error_sum(training_set);
        }
        error_sum
    }

    /**
    * Compute a new music
    * @input division : The music's division
    * @input nb_ticks : The number of ticks
    * @return The computed music
    */
    pub fn compute_music(&mut self, division : f64, nb_ticks : usize) -> CesureMusic {
        let infos = Matrix::new_row_from_datas(vec![division/DIVISION_RANGE, (nb_ticks as f64)/NB_TICKS_RANGE]);
        self.new_sequence(&infos);
        let mut output = CesureMusic {
            infos : infos.clone(),
            chords : Vec::new(),
            min_octave : 4,
        };

        for i in 0..nb_ticks {
            output.chords.push(self.compute_next());
        }

        output
    }
    pub fn compute_music_from_infos(&mut self, infos : &Matrix, nb_ticks : usize) -> CesureMusic {
        self.new_sequence(&infos);
        let mut output = CesureMusic {
            infos : infos.clone(),
            chords : Vec::new(),
            min_octave : 4,
        };

        for i in 0..nb_ticks {
            output.chords.push(self.compute_next());
        }

        output
    }

    pub fn train_from_midis_in_folder(&mut self, magnitude0 : f64, magnitude1 : f64, iterations : usize) {
        let paths = fs::read_dir("./").unwrap();

        let mut midi_paths = Vec::new();
        for path in paths {
            let path = path.unwrap().path();
            print!("{}", path.display());
            if path.display().to_string().ends_with(".mid") {
                midi_paths.push(path);
                println!(" is midi");
            } else {
                println!(" is not midi");
            }
        }

        assert!(midi_paths.len() > 0);
        println!("Found {} midi files to train on!", midi_paths.len());

        let mut musics = Vec::with_capacity(midi_paths.len());
        for path in &midi_paths {
            musics.push(CesureMusic::from_path(path));
        }

        let mut training_sets = Vec::with_capacity(musics.len());
        for music in &musics {
            training_sets.push(music.to_training_set(15));
        }

        genetic::train_multi(self, magnitude0, magnitude1, iterations, &training_sets);

    }

}