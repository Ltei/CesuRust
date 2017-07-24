
extern crate crossbeam;

use std::str;
use std::io::{self, Read, BufRead};
use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc::channel;
use std::thread;
use std::sync::{Arc, Mutex};

use utils::string::is_control;


pub struct DoubleI32Channel {
    pub sender : Sender<i32>,
    pub receiver : Receiver<i32>,
}
impl DoubleI32Channel {
    pub fn new() -> (DoubleI32Channel, DoubleI32Channel) {
        let (s1, r1) = channel();
        let (s2, r2) = channel();
        let c1 = DoubleI32Channel{sender: s1, receiver: r2};
        let c2 = DoubleI32Channel{sender: s2, receiver: r1};
        (c1, c2)
    }
}

pub trait AsyncStdinRead {
    fn read_line(&mut self) -> Option<String>;
    fn read_line_blocking(&mut self) -> String;
}
pub struct AsyncStdinReader {
    started_reading : bool,
    finished_reading : bool,
    buffer : Vec<u8>,
}
impl AsyncStdinReader {
    pub fn new() -> Arc<Mutex<AsyncStdinReader>> {
        let reader = AsyncStdinReader{started_reading: false, finished_reading: false, buffer: Vec::new()};
        Arc::new(Mutex::new(reader))
    }
}
impl AsyncStdinRead for Arc<Mutex<AsyncStdinReader>> {
    fn read_line(&mut self) -> Option<String> {

        let mut main_unwrap = self.lock().unwrap();

        if main_unwrap.started_reading {
            if main_unwrap.finished_reading {
                let line = str::from_utf8(main_unwrap.buffer.as_slice()).expect("Error").to_string();
                main_unwrap.buffer.clear();
                main_unwrap.started_reading = false;
                main_unwrap.finished_reading = false;
                return Some(line);
            }
        } else {
            let this = self.clone();

            thread::spawn(move || {
                let mut buffer = &mut [0];
                let stdin = io::stdin();
                let mut handle = stdin.lock();

                loop {
                    handle.read(buffer).expect("Error reading stdin");
                    let str = str::from_utf8(buffer).expect("Error");
                    if str == "\n" {
                        break
                    } else if !is_control(str) {
                        let mut thread_unwrap = this.lock().unwrap();
                        thread_unwrap.buffer.push(buffer[0]);
                    }
                }

                let mut thread_unwrap = this.lock().unwrap();
                thread_unwrap.finished_reading = true;
            });

            main_unwrap.started_reading = true;
        }

        return None;
    }
    fn read_line_blocking(&mut self) -> String {
        let mut line = self.read_line();
        while line.is_none() {
            line = self.read_line();
        }
        line.unwrap()
    }
}


pub fn stdin_readline() -> String {
    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    handle.read_line(&mut buffer).expect("Error reading line");
    let mut line = String::new();
    for str in buffer.as_str().split_whitespace() {
        line.push_str(str);
    }
    return line;
}