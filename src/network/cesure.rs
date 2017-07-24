
extern crate rand;

use std::clone::Clone;
use std::ops::Add;
use std::fs;

use network::gate::activation::Activation;
use network::gate::activation;
use network::gate::feedforward_gate;
use network::gate::feedforward_gate::FeedforwardGate;
use network::training::training_set::TrainingSet;
use network::training::error_calculation::ErrorCalculation;
use network::training::error_calculation;
use network::training::genetic;
use network::training::backpropagation;

use network::music::CesureMusic;
use network::music::DIVISION_RANGE;
use network::music::NB_TICKS_RANGE;
use network::music::MIN_KEY_RANGE;
use network::music::INFOS_DIMENSION;
use network::music::CHORD_DIMENSION;

use utils::matrix::Matrix;
use utils::matrix_math::row_concatenate;
use utils::traits::Parse;



pub struct Cesure {
    pub infos_dimension : usize,
    pub context_dimension : usize,
    pub output_dimension : usize,
    pub output_gate : FeedforwardGate,
    pub memory_gate : FeedforwardGate,
    pub infos : Matrix,
    pub context : Matrix,
}

pub struct VerboseOutput {
    pub output_out : feedforward_gate::VerboseOutput,
    pub memory_out : feedforward_gate::VerboseOutput,
}



impl Clone for Cesure {
    fn clone(&self) -> Cesure {
        return Cesure {
            infos_dimension : self.infos_dimension,
            context_dimension : self.context_dimension,
            output_dimension : self.output_dimension,
            output_gate : self.output_gate.clone(),
            memory_gate : self.memory_gate.clone(),
            infos : self.infos.clone(),
            context : self.context.clone(),
        }
    }
    fn clone_from(&mut self, source: &Cesure) {
        self.infos_dimension = source.infos_dimension;
        self.context_dimension = source.context_dimension;
        self.output_dimension = source.output_dimension;
        self.output_gate.clone_from(&source.output_gate);
        self.memory_gate.clone_from(&source.memory_gate);
        self.infos.clone_from(&source.infos);
        self.context.clone_from(&source.context);
    }
}
impl Parse for Cesure {
    fn to_string(&self) -> String {
        let mut output = format!("{} {} {}", self.infos_dimension, self.context_dimension, self.output_dimension);
        output = output.add("\nOUTPUT_GATE\n");
        output = output.add(self.output_gate.to_string().as_str());
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

        let gates_str : Vec<&str> = str.split("\nOUTPUT_GATE\n").collect();
        assert!(gates_str.len() == 2);
        let gates_str : Vec<&str> = gates_str[1].split("\nMEMORY_GATE\n").collect();
        assert!(gates_str.len() == 2);

        let output_gate = FeedforwardGate::from_string(gates_str[0]);
        let memory_gate = FeedforwardGate::from_string(gates_str[1]);

        return Cesure {
            infos_dimension : infos_dimension,
            context_dimension : context_dimension,
            output_dimension : output_dimension,
            output_gate : output_gate,
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
    pub fn new(context_dimension : usize) -> Cesure {
        let infos_context_dimension = INFOS_DIMENSION + context_dimension;
        let infos_context_output_dimension = infos_context_dimension + CHORD_DIMENSION;
        return Cesure {
            infos_dimension : INFOS_DIMENSION,
            context_dimension : context_dimension,
            output_dimension : CHORD_DIMENSION,
            //output_gate : FeedforwardGate::new2(infos_context_dimension, CHORD_DIMENSION, hidden_layers_dimensions.clone(), Activation::new(&ActivationType::Sigmoid)),
            //memory_gate : FeedforwardGate::new2(infos_context_output_dimension, context_dimension, hidden_layers_dimensions.clone(), Activation::new(&ActivationType::Sigmoid)),
            output_gate : FeedforwardGate::new_auto(infos_context_dimension, CHORD_DIMENSION, 10, Activation::new(activation::TYPE_SIGMOID)),
            memory_gate : FeedforwardGate::new_auto(infos_context_output_dimension, context_dimension, 10, Activation::new(activation::TYPE_SIGMOID)),
            infos : Matrix::new_row(INFOS_DIMENSION),
            context : Matrix::new_row(context_dimension),
        }
    }

    /**
    * Return a cloned object with some random changes
    * @input magnitude : The amount of random to apply
    * @input rand : The ThreadRng object to use for random
    * @return The cloned Cesure object
    */
    pub fn clone_randomized(&self, magnitude : f64, rand : &mut rand::ThreadRng) -> Cesure {
        return Cesure {
            infos_dimension : self.infos_dimension,
            context_dimension : self.context_dimension,
            output_dimension : self.output_dimension,
            output_gate : self.output_gate.clone_randomized(magnitude, rand),
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
        println!("- Output Gate :");
        self.output_gate.print();
        println!("- Memory Gate :");
        self.memory_gate.print();
    }

    /**
    * Return the number of neurons in this network
    * @return The number of neurons
    */
    pub fn get_nb_neurons(&self) -> usize {
        self.output_gate.get_nb_neurons() + self.memory_gate.get_nb_neurons()
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
        let infos_context = row_concatenate(&self.infos, &self.context);
        let output = self.output_gate.compute(&infos_context);

        let infos_context_output = row_concatenate(&infos_context, &output);
        let memory_out = self.memory_gate.compute(&infos_context_output);
        self.context = memory_out;

        output
    }

    /**
    * Compute the next output of the current sequence, and return
    * a VerboseOutput object, containing all the computing infos
    * @return The VerboseOutput object
    */
    pub fn compute_next_verbose(&mut self) -> VerboseOutput {
        let infos_context = row_concatenate(&self.infos, &self.context);
        let output = self.output_gate.compute_verbose(&infos_context);

        let infos_context_output = row_concatenate(&infos_context, &output.output);
        let memory_out = self.memory_gate.compute_verbose(&infos_context_output);
        self.context.clone_from(&memory_out.output);

        return VerboseOutput {
            output_out : output,
            memory_out : memory_out,
        }
    }

    /**
    * Manually inject the next output, changing the context
    * @input input : The input to inject
    */
    pub fn inject_next(&mut self, input : &Matrix) {
        let mut infos_context_output = row_concatenate(&self.infos, &self.context);
        infos_context_output.row_concatenate(&input);
        let memory_out = self.memory_gate.compute(&infos_context_output);
        self.context = memory_out;
    }

    /**
    * Return the error of this network, on a TrainingSet
    * @input training_set : The TrainingSet object to calculate the error on
    * @return The error
    */
    pub fn calculate_error_sum(&mut self, training_set : &TrainingSet, error_calculation: &ErrorCalculation) -> f64 {
        self.new_sequence(&training_set.infos);
        for i in 0..training_set.inject_sequence.len() {
            self.inject_next(&training_set.inject_sequence[i]);
        }
        let mut error_sum = 0.0;
        for i in 0..training_set.compute_sequence.len() {
            let output = self.compute_next();
            let error = (error_calculation.calculate)(&output, &training_set.compute_sequence[i]);
            let error_avg = error.get_abs_avg();
            error_sum += error_avg;
            /*output.print_title(format!("#{}, Output :", i).as_str());
            training_set.compute_sequence[i].print_title(format!("#{}, Ideal :", i).as_str());
            error.print_title("Error :");
            println!("Error avg : {}", error_avg);*/
        }
        error_sum
    }

    /**
    * Return the error of this network, on a Vec<TrainingSet>
    * @input training_set : The TrainingSet object to calculate the error on
    * @return The error
    */
    pub fn calculate_error_sum_multi(&mut self, training_sets : &Vec<TrainingSet>, error_calculation: &ErrorCalculation) -> f64 {
        let mut error_sum = 0.0;
        for training_set in training_sets {
            error_sum += self.calculate_error_sum(training_set, error_calculation);
        }
        error_sum
    }

    /**
    * Compute a music from a infos Matrix
    * @input infos : A matrix representing the music infos
    * @input inject_sequence : The notes sequence to inject before computing
    * @input nb_ticks : The music number of ticks
    * @return The computed music
    */
    pub fn compute_music_from_infos(&mut self, infos : &Matrix, inject_sequence: &Vec<Matrix>, nb_ticks : usize) -> CesureMusic {
        self.new_sequence(&infos);
        let mut output = CesureMusic {
            infos : infos.clone(),
            chords : Vec::new(),
        };

        for i in 0..inject_sequence.len() {
            self.inject_next(&inject_sequence[i]);
            output.chords.push(inject_sequence[i].clone());
        }
        for _ in 0..nb_ticks {
            let mut out = self.compute_next();
            CesureMusic::normalize_chord(&mut out);
            output.chords.push(out);
        }

        output
    }

    /**
    * Compute a music from a infos Matrix
    * @input division : The music division
    * @input nb_ticks : The music number of ticks
    * @input min_key : The minimum note key
    * @return The computed music
    */
    pub fn compute_music(&mut self, division: f64, nb_ticks: usize, min_key: usize) -> CesureMusic {
        let infos = Matrix::new_row_from_datas(vec![division/DIVISION_RANGE, (nb_ticks as f64)/NB_TICKS_RANGE, (min_key as f64)/MIN_KEY_RANGE]);
        self.new_sequence(&infos);
        let mut output = CesureMusic {
            infos : infos.clone(),
            chords : Vec::new(),
        };

        for _ in 0..nb_ticks {
            let mut out = self.compute_next();
            CesureMusic::normalize_chord(&mut out);
            output.chords.push(out);
        }

        output
    }

    pub fn train_genetic_from_folder(&mut self, magnitude0 : f64, magnitude1 : f64, iterations : usize) {
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

        let error_calc = ErrorCalculation::new(error_calculation::ERROR_CALCULATION_TYPE_SMART);
        genetic::train(self, &training_sets, &error_calc, magnitude0, magnitude1, iterations);

    }
    pub fn train_backpropagation_from_folder(&mut self, learning_rate : f64, momentum : f64, iterations : usize) {
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

        let error_calculation = ErrorCalculation::new(error_calculation::ERROR_CALCULATION_TYPE_SMART);
        backpropagation::train(self, &training_sets, &error_calculation, learning_rate, momentum, iterations);
    }

    /**
    * A method for automatically train on the foler's midi files
    * The computing a music and saving the network and the computed
    * music
    */
    pub fn train_n_save(&mut self) {
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

        let error_calc = ErrorCalculation::new(error_calculation::ERROR_CALCULATION_TYPE_SMART);
        for _ in 0..1 {
            //backpropagation::train(self, &training_sets, &error_calc, 0.1, 0.9, 1000);
            backpropagation::train_mod2(self, &training_sets, &error_calc, 0.1, 0.9, 5, 100);
            //genetic::train(self, &training_sets, &error_calc, 1.0, 0.0, 100);
            //genetic::train_mod2(self, &training_sets, &error_calc, 0.1, 0.0, 10000);
        }

        self.compute_music_from_infos(&training_sets[0].infos, &training_sets[0].inject_sequence, 1000).save("output_test.mid");
        //self.save("cesure_test.ces");
    }

}


/*/**
    * Compute the next output of the current sequence and compute
    * it's error compared to the ideal chord
    * @return A tuple with the computed output and error
    */
    pub fn compute_next_and_calculate_error(&mut self, ideal_chord : &Matrix) -> (Matrix,f64) {
        let infos_context = row_concatenate(&self.infos, &self.context);
        let mut output = self.output_gate.compute(&infos_context);

        let mut error = 0.0;
        for i in 0..CHORD_DIMENSION {
            let mut note_error = abs(output.datas[i]-ideal_chord.datas[i]);
            if ideal_chord.datas[i] == 1.0 && output.datas[i] <= 0.75 {
                note_error *= 6.0;
            } else if ideal_chord.datas[i] == 0.0 && output.datas[i] > 0.75 {
                note_error *= 2.0;
            }
            error += note_error;
        }

        let infos_context_output = row_concatenate(&infos_context, &output);
        let memory_out = self.memory_gate.compute(&infos_context_output);
        self.context = memory_out;

        (output,error)
    }*/
/*fn normalize_output(chord : &mut Matrix) {
    assert!(chord.is_row() && chord.len == CHORD_DIMENSION);
    for i in 0..chord.len {
        if chord.datas[i] > 0.75 {
            chord.datas[i] = 1.0;
        } else {
            chord.datas[i] = 0.0;
        }
    }
}*/