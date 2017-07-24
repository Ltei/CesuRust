use std::ops::Add;
use std::path::Path;

use rimd::{SMF,SMFError,Event,TrackEvent,Track,SMFBuilder,MidiMessage, MetaEvent,SMFWriter};

use networkV2::training::training_set::TrainingSet;

use utils::matrix::Matrix;
use utils::traits::Parse;

pub const NB_OCTAVES : usize = 3;
pub const CHORD_DIMENSION : usize = 12*NB_OCTAVES;

pub const INFOS_DIMENSION : usize = 2;

pub const DIVISION_RANGE : f64 = 1000.0;
pub const NB_TICKS_RANGE : f64 = 1000000.0;



pub struct CesureMusic {
    pub infos : Matrix,
    pub chords : Vec<Matrix>,
    pub min_octave : usize,
}


/*impl Parse for CesureMusic {
    fn to_string(&self) -> String {
        let mut output = self.infos.to_string();
        for i in 0..self.chords.len() {
            output = output.add(format!("\n{}", self.chords[i].to_string()).as_str());
        }
        return output;
    }
    fn from_string(str : &str) -> CesureMusic {
        let lines : Vec<&str> = str.split("\n").collect();

        let infos = Matrix::from_string(lines[0]);
        assert!(infos.len == INFOS_DIMENSION);

        let mut chords = Vec::with_capacity(lines.len()-1);
        for i in 1..lines.len() {
            let chord = Matrix::from_string(lines[i]);
            assert!(chord.len == CHORD_DIMENSION);
            chords.push(chord);
        }

        return CesureMusic {
            infos : infos,
            chords : chords,
            min_octave : min_octave,
        }
    }
}*/


impl CesureMusic {

    pub fn from_path_str(file_path : &str) -> CesureMusic {
        return CesureMusic::from_path(&Path::new(file_path));
    }
    pub fn from_path(file_path : &Path) -> CesureMusic {
        match SMF::from_file(file_path) {
            Ok(smf) => {
                return CesureMusic::from_smf(&smf);
            }
            Err(e) => {
                println!("Error reading {}", file_path.display());
                match e {
                    SMFError::InvalidSMFFile(s) => {println!("{}",s);}
                    SMFError::Error(e) => {println!("io: {}",e);}
                    SMFError::MidiError(e) => {println!("Midi Error: {}",e);}
                    SMFError::MetaError(_) => {println!("Meta Error");}
                }
                loop {}
            }
        }
    }
    pub fn from_smf(smf : &SMF) -> CesureMusic {
        assert!(smf.tracks.len() == 2);
        let division = smf.division;

        let tick_div = (division / 4) as u64;
        let track = &smf.tracks[1];
        let parsed_notes = track_to_parsed_notes(track, tick_div);


        let nb_ticks = parsed_notes[parsed_notes.len()-1].tick;
        let mut min_octave = 10;

        for note in &parsed_notes {
            let octave = note.key / 12;
            if octave < min_octave {
                min_octave = octave;
            }
        }

        let infos = Matrix::new_row_from_datas(vec![(division as f64)/DIVISION_RANGE, (nb_ticks as f64)/NB_TICKS_RANGE]);
        let mut chords = Vec::with_capacity(nb_ticks as usize);
        for _ in 0..nb_ticks {
            chords.push(Matrix::new_row(CHORD_DIMENSION));
        }

        for note_i in 0..parsed_notes.len() {
            let note = &parsed_notes[note_i];
            if note.on == true {
                let key = note.key;
                let on_tick = note.tick;
                let mut off_i = note_i+1;
                while parsed_notes[off_i].key != key {
                    off_i += 1;
                }
                //assert!(parsed_notes[off_i].on == false);
                for i in on_tick..parsed_notes[off_i].tick {
                    let actual_key = key-(12*min_octave);
                    if actual_key >= CHORD_DIMENSION {
                        println!("Found out of bound key, deleting it...");
                    } else {
                        chords[i].datas[actual_key] = 1.0;
                    }
                }
            }
        }

        return CesureMusic {
            infos : infos,
            chords : chords,
            min_octave : min_octave,
        }
    }

    pub fn normalize_best(&mut self) {
        for chord_i in 0..self.chords.len() {
            let chord = &mut self.chords[chord_i];
            let mut best_value = 0.0;
            let mut best_index = 0;
            for i in 0..chord.len {
                if chord.datas[i] > best_value {
                    best_value = chord.datas[i];
                    best_index = i;
                    break;
                }
            }
            chord.set_zero();
            chord.datas[best_index] = 1.0;
        }
    }
    pub fn save(&self, file_path : &str) {

        let mut builder = SMFBuilder::new();

        let division = self.infos.datas[0] * DIVISION_RANGE;
        let tick_mult = division as u64 / 4;

        builder.add_track();
        let tempo = MetaEvent::tempo_setting(500000);
        let tsign = MetaEvent::time_signature(2, 2, 7, 160);
        let eot = MetaEvent::end_of_track();
        builder.add_event(0, TrackEvent{vtime: 0, event: Event::Meta(tempo)});
        builder.add_event(0, TrackEvent{vtime: 0, event: Event::Meta(tsign)});
        builder.add_event(0, TrackEvent{vtime: 1, event: Event::Meta(eot)});

        builder.add_track();
        for note_i in 0..CHORD_DIMENSION {
            let key = (note_i+12*self.min_octave) as u8;
            let mut on = false;
            for tick_i in 0..self.chords.len() {
                if on == false {
                    if self.chords[tick_i].datas[note_i] == 1.0 {
                        let note_on = MidiMessage::note_on(key, 100, 0);
                        builder.add_midi_abs(1, (tick_i as u64)*tick_mult, note_on);
                        on = true;
                    }
                } else {
                    if self.chords[tick_i].datas[note_i] == 0.0 {
                        let note_off = MidiMessage::note_off(key, 100, 0);
                        builder.add_midi_abs(1, tick_i as u64 * tick_mult, note_off);
                        on = false;
                    }
                }
            }
            if on {
                let note_off = MidiMessage::note_off(key, 100, 0);
                builder.add_midi_abs(1, self.chords.len() as u64 * tick_mult, note_off);
            }
        }
        let eot = MetaEvent::end_of_track();
        builder.add_event(1, TrackEvent{vtime: 1, event: Event::Meta(eot)});

        /*let smf = builder.result();
        let mut tnum = 1;
        for track in smf.tracks.iter() {
            let mut time: u64 = 0;
            println!("\n{}: {}\nevents:",tnum,track);
            tnum+=1;
            for event in track.events.iter() {
                println!("  {}",event.fmt_with_time_offset(time));
                time += event.vtime;
            }
        }*/

        let mut smf = builder.result();
        smf.division = division as i16;
        let writer = SMFWriter::from_smf(smf);
        writer.write_to_file(Path::new(file_path)).expect("Failed to write to file");

    }

    pub fn to_training_set(&self, nb_first_note_to_inject : usize) -> TrainingSet {
        assert!(nb_first_note_to_inject < self.chords.len());
        let mut training_set = TrainingSet {
            infos : self.infos.clone(),
            inject_sequence : Vec::new(),
            compute_sequence : Vec::new(),
        };
        for i in 0..nb_first_note_to_inject {
            training_set.inject_sequence.push(self.chords[i].clone());
        }
        for i in nb_first_note_to_inject..self.chords.len() {
            training_set.compute_sequence.push(self.chords[i].clone());
        }
        training_set
    }

}

struct ParsedNote {
    on : bool,
    tick : usize,
    key : usize,
}
fn track_to_parsed_notes(track : &Track, tick_div : u64) -> Vec<ParsedNote> {
    let mut parsed_notes = Vec::new();
    let mut tick = 0;
    for event in track.events.iter() {
        let message : String = event.fmt_with_time_offset(0);
        let parsed : Vec<&str> = message.as_str().split_whitespace().collect();
        if parsed.len() == 7 && parsed[2].eq("Note") {
            tick += event.vtime / tick_div;

            let mut on = false;
            if parsed[3].eq("On:") {
                on = true;
            } else if !parsed[3].eq("Off:") {
                panic!("wtf");
            }

            let note : Vec<&str> = parsed[4].split(",").collect();
            let key : usize = note[0].split_at(1).1.parse().expect("wtf");

            parsed_notes.push(ParsedNote {
                on : on,
                tick : tick as usize,
                key : key,
            });
        }
    }
    parsed_notes
}