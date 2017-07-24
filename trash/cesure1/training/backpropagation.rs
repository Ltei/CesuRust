
use network::utils::feedforward_gate;
use network::cesure1::cesure;
use network::cesure1::cesure::Cesure;
use utils::matrix::Matrix;
use utils::matrix_math::row_concatenate;
use utils::matrix_math::p_mult;
use utils::matrix_math::sub;
use utils::matrix_math::row_slice;
use utils::matrix_math::row_merge_avg;



pub struct CesureVerboseOutput {
    pub context_before : Matrix,
    pub output_out : feedforward_gate::VerboseOutput,
    pub forget_out : feedforward_gate::VerboseOutput,
    pub memory_out : feedforward_gate::VerboseOutput,
    pub input_out : feedforward_gate::VerboseOutput,
}





#[allow(dead_code)]
pub fn train (cesure : &mut Cesure, infos : &Matrix, input_sequence : &Matrix, compute_sequence : &Matrix) -> f64 {
    cesure.reset_context();
    cesure.set_infos(&infos);

    for i in 0..input_sequence.rows {
        cesure.input_next(&input_sequence.get_row(i));
    }


    let mut error_sum = 0.0;
    let sequence_len = compute_sequence.rows;
    let mut outputs = Vec::with_capacity(sequence_len);
    for i in 0..sequence_len {
        outputs.push( cesure_compute_next_verbose(cesure) );
        let error = sub(&outputs[i].output_out.output, &compute_sequence.get_row(i));
        error_sum += error.clone().abs().get_avg();
        let signal = cesure.output_gate.backpropagate_error_signal(&outputs[i].output_out, &error, 1.0);
        if i > 1 {
            let mut context_signal = row_slice(&signal, cesure::INFOS_DIMENSION).1;
            let mut last_i = i - 1;
            loop {
                let input_signal = p_mult(&context_signal, &outputs[last_i].memory_out.output);
                let memory_signal = p_mult(&context_signal, &outputs[last_i].input_out.output);
                let forget_signal = p_mult(&context_signal, &outputs[last_i].context_before);

                let input_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].input_out, &input_signal, 1.0);
                let memory_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].memory_out, &memory_signal, 1.0);
                let forget_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].forget_out, &forget_signal, 1.0);

                let mut merged_signal = row_merge_avg(&input_signal, &row_merge_avg(&memory_signal, &forget_signal));
                merged_signal = row_slice(&merged_signal, cesure::INFOS_DIMENSION+cesure::CONTEXT_DIMENSION).0;
                merged_signal = row_slice(&merged_signal, cesure::INFOS_DIMENSION).1;
                context_signal = row_merge_avg(&context_signal, &merged_signal);

                if last_i > 0 {
                    last_i -= 1;
                } else {
                    break;
                }
            }
        }
    }

    error_sum
}


#[allow(dead_code)]
pub fn cesure_compute_next_verbose(cesure : &mut Cesure) -> CesureVerboseOutput {
    let context_before = cesure.context.clone();
    let infos_context = row_concatenate(&cesure.infos, &cesure.context);
    let output = cesure.output_gate.compute_verbose(&infos_context);
    let infos_context_output = row_concatenate(&infos_context, &output.output);
    let forget_out = cesure.forget_gate.compute_verbose(&infos_context_output);
    let memory_out = cesure.memory_gate.compute_verbose(&infos_context_output);
    let input_out = cesure.input_gate.compute_verbose(&infos_context_output);
    cesure.context.p_mult(&forget_out.output);
    cesure.context.add(&p_mult(&memory_out.output,&input_out.output));
    return CesureVerboseOutput {
        context_before : context_before,
        output_out : output,
        forget_out : forget_out,
        memory_out : memory_out,
        input_out : input_out,
    }
}







/*#[allow(dead_code)]
pub fn train (cesure : &mut Cesure, infos : &Matrix, sequence : &Matrix) -> f64 {
    let mut error_sum = 0.0;
    let sequence_len = sequence.rows;
    let mut outputs = Vec::with_capacity(sequence_len);
    for i in 0..sequence_len {
        outputs.push( cesure_compute_next_verbose(cesure) );
        let error = sub(&outputs[i].output_out.output, &sequence.get_row(i));
        error_sum += error.clone().abs().get_avg();
        let signal = cesure.output_gate.backpropagate_error_signal(&outputs[i].output_out, &error, 1.0);
        if i > 1 {
            let mut context_signal = row_slice(&signal, cesure::INFOS_DIMENSION).1;
            let mut lastI = i - 1;
            loop {
                let input_signal = p_mult(&context_signal, &outputs[lastI].memory_out.output);
                let memory_signal = p_mult(&context_signal, &outputs[lastI].input_out.output);
                let forget_signal = p_mult(&context_signal, &outputs[lastI].context_before);
                cesure.input_gate.backpropagate_error_signal(&outputs[lastI].input_out, &input_signal, 1.0);
                cesure.input_gate.backpropagate_error_signal(&outputs[lastI].memory_out, &memory_signal, 1.0);
                cesure.input_gate.backpropagate_error_signal(&outputs[lastI].forget_out, &forget_signal, 1.0);
                if lastI > 0 {
                    lastI -= 1;
                } else {
                    break;
                }
            }
        }
    }

    error_sum
}*/


/*#[allow(dead_code)]
pub fn train (cesure : &mut Cesure, infos : &Matrix, sequence : &Matrix) -> f64 {
    cesure.reset_context();
    cesure.set_infos(&infos);

    let mut error_sum = 0.0;
    let sequence_len = sequence.rows;
    let mut outputs = Vec::with_capacity(sequence_len);
    for i in 0..sequence_len {
        outputs.push( cesure_compute_next_verbose(cesure) );
        let error = sub(&outputs[i].output_out.output, &sequence.get_row(i));
        error_sum += error.clone().abs().get_avg();
        let signal = cesure.output_gate.backpropagate_error_signal(&outputs[i].output_out, &error, 1.0);
        if i > 1 {
            let mut context_signal = row_slice(&signal, cesure::INFOS_DIMENSION).1;
            let mut last_i = i - 1;
            loop {
                let input_signal = p_mult(&context_signal, &outputs[last_i].memory_out.output);
                let memory_signal = p_mult(&context_signal, &outputs[last_i].input_out.output);
                let forget_signal = p_mult(&context_signal, &outputs[last_i].context_before);

                let input_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].input_out, &input_signal, 1.0);
                let memory_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].memory_out, &memory_signal, 1.0);
                let forget_signal = cesure.input_gate.backpropagate_error_signal(&outputs[last_i].forget_out, &forget_signal, 1.0);

                let mut merged_signal = row_merge_avg(&input_signal, &row_merge_avg(&memory_signal, &forget_signal));
                merged_signal = row_slice(&merged_signal, cesure::INFOS_DIMENSION+cesure::CONTEXT_DIMENSION).0;
                merged_signal = row_slice(&merged_signal, cesure::INFOS_DIMENSION).1;
                context_signal = row_merge_avg(&context_signal, &merged_signal);

                if last_i > 0 {
                    last_i -= 1;
                } else {
                    break;
                }
            }
        }
    }

    error_sum
}*/