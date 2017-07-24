#![allow(dead_code)]

use utils::matrix::Matrix;

use networkV2::cesure;

use networkV2::music::INFOS_DIMENSION;
use networkV2::music::CHORD_DIMENSION;

pub struct TrainingSet {
    pub infos : Matrix,
    pub inject_sequence : Vec<Matrix>,
    pub compute_sequence : Vec<Matrix>,
}


impl TrainingSet {

    pub fn new(infos : Matrix, inject_sequence : Vec<Matrix>, compute_sequence : Vec<Matrix>) -> TrainingSet {
        assert!(infos.is_row() && infos.len == INFOS_DIMENSION);
        for i in 0..inject_sequence.len() {
            assert!(inject_sequence[i].is_row() && inject_sequence[i].len == CHORD_DIMENSION);
        }
        for i in 0..compute_sequence.len() {
            assert!(compute_sequence[i].is_row() && compute_sequence[i].len == CHORD_DIMENSION);
        }
        return TrainingSet {
            infos : infos,
            inject_sequence : inject_sequence,
            compute_sequence : compute_sequence,
        }
    }

}