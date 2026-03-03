use anyhow::{anyhow, bail, Context};
use std::io::Write;
use std::path::{Path, PathBuf};

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

struct ArgsNormal {
    in_path: PathBuf,
    out_path: PathBuf,
    min_rate: u32,
    base_note: u8,
}

enum Args {
    Normal(ArgsNormal),
    Help
}

fn parse_base_note(note: &str) -> Result<u8, anyhow::Error> {
    let end_of_note_name = note.find(|c: char| {
        c.is_ascii_digit() || c == '-'
    }).ok_or(anyhow!("No octave number found in note name '{}'", note))?;
    
    let (note_name_str, octave_str) = note.split_at(end_of_note_name);
    let mut midi_note: i16 = match note_name_str {
        "C" | "c" => 0,
        "C#" | "c#" | "Db" | "db" => 1,
        "D" | "d" => 2,
        "D#" | "d#" | "Eb" | "eb" => 3,
        "E" | "e" => 4,
        "F" | "f" => 5,
        "F#" | "f#" | "Gb" | "gb" => 6,
        "G" | "g" => 7,
        "G#" | "g#" | "Ab" | "ab" => 8,
        "A" | "a" => 9,
        "A#" | "a#" | "Bb" | "bb" => 10,
        "B" | "b" => 11,
        _ => bail!("Invalid note name '{}'", note_name_str),
    };
    let octave_int: i16 = octave_str.parse()
        .with_context(|| format!("Couldn't convert '{}' to integer", octave_str))?;
    midi_note += (octave_int + 1i16) * 12;
    midi_note.try_into().with_context(|| format!("Calculated note ({}) out of byte range", midi_note))
}

fn parse_args(parser: &mut lexopt::Parser) -> Result<Args, anyhow::Error> {
    use lexopt::prelude::*;
    
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut min_rate = 10000u32;
    let mut base_note = 60u8;
    
    while let Some(arg) = parser.next()? {
        match arg {
            Short('h') | Short('?') | Long("help") => {
                return Ok(Args::Help);
            }
            Short('m') | Long("min-rate") => {
                min_rate = parser.value()?.parse()?;
            }
            Short('b') | Long("base") => {
                base_note = parser.value()?.parse_with(parse_base_note)?;
            }
            Value(val) if in_path.is_none() => {
                let val_path: &Path = val.as_ref();
                in_path = Some(val_path.to_path_buf());
            }
            Value(val) if out_path.is_none() => {
                let val_path: &Path = val.as_ref();
                out_path = Some(val_path.to_path_buf());
            }
            _ => return Err(arg.unexpected().into()),
        }
    }
    
    Ok(Args::Normal(ArgsNormal {
        in_path: in_path.ok_or(anyhow!("No input file path provided"))?,
        out_path: out_path.ok_or(anyhow!("No output file path provided"))?,
        min_rate,
        base_note,
    }))
}

fn print_usage(bin_name: &str) {
    let stderr = std::io::stderr();
    let mut handle = stderr.lock();
    // Ignore potential errors writing to stderr... I don't think anyone will
    // miss this message if they happen to redirect stderr to a file and fail at
    // converting a gazillion samples
    _ = writeln!(handle, "mp2kwav - WAV sample converter for GBA MusicPlayer2000 (BIOS sound engine) samples");
    _ = writeln!(handle, "Usage:");
    _ = writeln!(handle, "    {} [-h?|--help]", bin_name);
    _ = writeln!(handle, "    {} [-m|--min-rate=<number>] <gba-sample.bin> <output-wav-sample.wav>", bin_name);
    _ = writeln!(handle, "    {} [-b|--base=<note>] <wav-sample.wav> <output-gba-sample.bin>", bin_name);
    _ = writeln!(handle, "Options:");
    _ = writeln!(handle, "    -b, --base: Set base note for the sample when converting .wav --> .bin (default: C4)");
    _ = writeln!(handle, "    -m, --min-rate: Set permitted WAV sample rate to lie in the range [n, 2n) (default: 10000)");
    _ = writeln!(handle, "    -h, -?, --help: Display this help message");
    _ = writeln!(handle);
    _ = writeln!(handle, "See NOTICES.txt for copyright notices from all used libraries.");
}

fn main() -> Result<(), anyhow::Error> {
    let mut args_parser = lexopt::Parser::from_env();
    
    let args = parse_args(&mut args_parser)?;
    if let Args::Help = args {
        print_usage(args_parser.bin_name().unwrap_or("mp2kwav"));
        return Ok(());
    }
    let Args::Normal(an) = args else { unreachable!() };
    let in_path = an.in_path;
    let out_path = an.out_path;
    let min_rate = an.min_rate;
    let base_note = an.base_note;
    
    if let Some(in_ext) = in_path.extension() {
        // Load all data from in_path
        let infile_data = std::fs::read(&in_path)
            .with_context(|| format!("Failed to read sample from {:?}", in_path))?;
        
        if in_ext == Path::new("wav") {
            // Convert to GBA format
            let gba_sample = mp2kwav::sample::convert_to_gba(&infile_data[..], base_note)
                .with_context(|| format!("Failed to convert sample data from file {:?}", in_path))?;
            
            std::fs::write(&out_path, gba_sample)
                .with_context(|| format!("Failed to write BIN to {:?}", out_path))?;
        } else if in_ext == Path::new("bin") {
            // Convert to WAV format
            let (wav_data, base_note) = mp2kwav::sample::convert_to_wav(&infile_data[..], min_rate)
                .with_context(|| format!("Failed to convert sample data from file {:?}", in_path))?;
            
            println!("Base note: {}{}", NOTE_NAMES[usize::from(base_note) % 12], base_note / 12 - 1);
            
            std::fs::write(&out_path, wav_data)
                .with_context(|| format!("Failed to write WAV to {:?}", out_path))?;
        } else {
            bail!("Unrecognized input file extension '{:?}' -- expected 'bin' or 'wav'", in_ext);
        }
    } else {
        bail!("Input file name has no file extension. Please rename the file to <name>.wav or <name>.bin as appropriate.");
    }
    Ok(())
}
