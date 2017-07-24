#![allow(dead_code)]
#![allow(unused_imports)]

extern crate rand;
extern crate time;
extern crate rimd;
extern crate crossbeam;

use std::time::Instant;

mod network;
mod utils;

use network::cesure::Cesure;
use network::music::CHORD_DIMENSION;
use utils::traits::Parse;

use utils::io::{AsyncStdinReader, AsyncStdinRead};


fn main() {

    /*let mut stdin = AsyncStdinReader::new();

    println!("Welcome to CesureTrainer!");
    println!("What do you want to do?");
    println!("- [0] Create new network");
    println!("- [1] Load network");
    let cesure = match stdin.read_line_blocking().as_str() {
        "0" => {
            println!("Enter the network's context dimension");
            let mut dimension = stdin.read_line_blocking().parse();
            while dimension.is_err() {
                println!("Enter a valid usize");
                dimension = stdin.read_line_blocking().parse();
            }
            Cesure::new(dimension.unwrap());
        },
        "1" => {
            println!("Enter the network's file name");
            Cesure::load(stdin.read_line_blocking().as_str());
        }
        _ => panic!("Fuck you.")
    };

    println!("What do you want to do?");
    println!("- [0] Create new network");
    println!("- [1] Load network");*/

    let mut cesure : Cesure = Cesure::load("cesure_test.ces");
    cesure.train_n_save();

}


/*fn m_dot_test() {
    let mut rand = rand::thread_rng();

    let mut row = Matrix::new_row(5);
    let mut col = Matrix::new_col(5);
    let mut square = Matrix::new(5,5);
    row.set_random_int(-5, 5, &mut rand);
    col.set_random_int(-5, 5, &mut rand);
    square.set_random_int(-5, 5, &mut rand);

    row.print_title("::: row :::");
    col.print_title("::: col :::");
    square.print_title("::: square :::");

    println!();

    m_dot(&row, &col).print_title("::: row dot col :::");
    m_dot(&row, &square).print_title("::: row dot square :::");
    m_dot(&square, &col).print_title("::: square dot col :::");

    println!();
    println!();

    m_dot_new(&row, &col).print_title("::: row dot_new col :::");
    m_dot_new(&row, &square).print_title("::: row dot_new square :::");
    m_dot_new(&square, &col).print_title("::: square dot_new col :::");
    let mut big_square_0 = Matrix::new(100,100);
    let mut big_square_1 = Matrix::new(100,100);
    big_square_0.set_random(-10.0, 10.0, &mut rand);
    big_square_1.set_random(-10.0, 10.0, &mut rand);

    let nb_iterations = 2500;

    let t0 = Instant::now();
    for i in 0..nb_iterations {
        m_dot(&big_square_0, &big_square_1);
    }
    let delta_t = t0.elapsed();
    println!("Performance for m_dot :");
    println!("{:?}", delta_t);
    println!("Elapsed: {} ms", (delta_t.as_secs() * 1_000) + (delta_t.subsec_nanos() / 1_000_000) as u64);

    let t0 = Instant::now();
    for i in 0..nb_iterations {
        m_dot_new(&big_square_0, &big_square_1);
    }
    let delta_t = t0.elapsed();
    println!("Performance for m_dot :");
    println!("{:?}", delta_t);
    println!("Elapsed: {} ms", (delta_t.as_secs() * 1_000) + (delta_t.subsec_nanos() / 1_000_000) as u64);
}*/