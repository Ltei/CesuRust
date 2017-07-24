
use std::path::Path;
use std::io::Read;
use std::io::Write;
use std::fs::File;
use std::fs::OpenOptions;


pub trait Parse : Sized {
    fn to_string(&self) -> String;
    fn from_string(str : &str) -> Self;

    fn save(&self, file_path : &str) {
        let path = Path::new(file_path);
        let mut file = OpenOptions::new().create(true).write(true)
            .open(path).expect("Couldn't open file");
        file.set_len(0).expect("Couldn't clear file");
        file.write(self.to_string().as_bytes()).expect("Couldn't write to file");
    }
    fn load(file_path : &str) -> Self {
        let path = Path::new(file_path);
        let mut file : File = File::open(path).expect("Couln't open file");
        let mut str = String::new();
        file.read_to_string(&mut str).expect("Couldn't read file");
        return Self::from_string(str.as_str());
    }
}