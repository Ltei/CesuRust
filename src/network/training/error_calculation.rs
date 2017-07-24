
use utils::matrix::Matrix;
use network::music::CHORD_DIMENSION;


pub const ERROR_CALCULATION_TYPE_BASIC : u8 = 0;
pub const ERROR_CALCULATION_TYPE_ONLY_ON : u8 = 1;
pub const ERROR_CALCULATION_TYPE_SMART : u8 = 2;


pub struct ErrorCalculation {
    pub calculation_type: u8,
    pub calculate: fn(output_chord: &Matrix, ideal_chord: &Matrix) -> Matrix,
}


impl Clone for ErrorCalculation {
    fn clone(&self) -> ErrorCalculation {
        ErrorCalculation {
            calculation_type : self.calculation_type,
            calculate : self.calculate,
        }
    }
    fn clone_from(&mut self, source: &ErrorCalculation) {
        self.calculation_type = source.calculation_type;
        self.calculate = source.calculate;
    }
}


impl ErrorCalculation {

    pub fn new(calculation_type: u8) -> ErrorCalculation {
        let error_calculation = match calculation_type {
            ERROR_CALCULATION_TYPE_BASIC => calculation_basic,
            ERROR_CALCULATION_TYPE_ONLY_ON => calculation_only_on,
            ERROR_CALCULATION_TYPE_SMART => calculation_smart,
            _ => panic!("Unknown calculation type"),
        };
        return ErrorCalculation {
            calculation_type : calculation_type,
            calculate : error_calculation,
        }
    }

}


fn calculation_basic(output_chord: &Matrix, ideal_chord: &Matrix) -> Matrix {
    assert!(output_chord.is_row() && output_chord.len == CHORD_DIMENSION);
    assert!(ideal_chord.is_row() && ideal_chord.len == CHORD_DIMENSION);
    let mut output = Vec::with_capacity(CHORD_DIMENSION);
    for i in 0..CHORD_DIMENSION {
        output.push(match ideal_chord.datas[i] {
            0.0 => output_chord.datas[i],
            1.0 => output_chord.datas[i] - 1.0,
            _ => panic!("Ideal chord isn't normalized"),
        });
    }
    Matrix::new_row_from_datas(output)
}

fn calculation_only_on(output_chord: &Matrix, ideal_chord: &Matrix) -> Matrix {
    assert!(output_chord.is_row() && output_chord.len == CHORD_DIMENSION);
    assert!(ideal_chord.is_row() && ideal_chord.len == CHORD_DIMENSION);
    let mut output = Vec::with_capacity(CHORD_DIMENSION);
    for i in 0..CHORD_DIMENSION {
        output.push(match ideal_chord.datas[i] {
            0.0 => 0.0,
            1.0 => output_chord.datas[i] - 1.0,
            _ => panic!("Ideal chord isn't normalized"),
        });
    }
    Matrix::new_row_from_datas(output)
}

fn calculation_smart(output_chord: &Matrix, ideal_chord: &Matrix) -> Matrix {
    assert!(output_chord.is_row() && output_chord.len == CHORD_DIMENSION);
    assert!(ideal_chord.is_row() && ideal_chord.len == CHORD_DIMENSION);
    let mut output = Vec::with_capacity(CHORD_DIMENSION);
    for i in 0..CHORD_DIMENSION {
        output.push(match ideal_chord.datas[i] {
            0.0 => {
                let error = {
                    let delta = output_chord.datas[i];
                    match delta > 0.9 {
                        true => 9.0*delta - 8.0,
                        false => delta/9.0,
                    }
                };
                error / 12.0
            },
            1.0 => {
                let error = output_chord.datas[i] - 1.0;
                error
            },
            _ => panic!("Ideal chord isn't normalized"),
        });
    }
    Matrix::new_row_from_datas(output)
}