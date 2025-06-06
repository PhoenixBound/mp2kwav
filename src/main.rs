use anyhow::{anyhow, bail, Context};
use std::path::PathBuf;
use std::sync::LazyLock;

const NOTE_NAMES: [&'static str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

static TWELFTH_ROOT_OF_TWO_POWERS: LazyLock<[f64; 12]> = LazyLock::new(|| {
    [
        1.,                (1f64/12.).exp2(), (2f64/12.).exp2(),         (3f64/12.).exp2(),
        (4f64/12.).exp2(), (5f64/12.).exp2(), core::f64::consts::SQRT_2, (7f64/12.).exp2(),
        (8f64/12.).exp2(), (9f64/12.).exp2(), (10f64/12.).exp2(),        (11f64/12.).exp2()
    ]
});

/// Add a stand-in for a function similar to IEEE-754 `remainder` operations.
/// The argument is always positive for our purposes so it should be equivalent
fn abs_remainder_by_one(lhs: f64) -> f64 {
    if lhs % 1.0 > 0.5 {
        1.0 - lhs % 1.0
    } else {
        lhs % 1.0
    }
}

fn guess_base_note_from_sample_pitch(sample_pitch: u32, min_rate: u32) -> Option<(u32, u8)> {
    // Start at middle C
    let mut base_note = 60i16;
    
    // We want to get one possible base note value, so restrict our valid range
    // to one octave
    let sample_rate_upper_bound = f64::from(min_rate) * 2.;
    let sample_rate_lower_bound = f64::from(min_rate);
    
    // Then, move into that octave range.
    let mut sample_rate: f64 = f64::from(sample_pitch) / 1024.;
    while sample_rate >= sample_rate_upper_bound {
        sample_rate /= 2.;
        base_note -= 12;
    }
    while sample_rate < sample_rate_lower_bound {
        sample_rate *= 2.;
        base_note += 12;
    }
    
    // Then figure out which of the 12 notes in the octave is closest to having
    // a whole-number sample rate
    // Starting with C
    let mut best_note_name = 0u8;
    let mut min_difference = abs_remainder_by_one(sample_rate);
    for i in 1u8..12 {
        let this_sample_rate = sample_rate * TWELFTH_ROOT_OF_TWO_POWERS[usize::from(i)];
        let score = abs_remainder_by_one(this_sample_rate);
        if score < min_difference {
            min_difference = score;
            best_note_name = i;
        }
    }
    
    let mut best_sample_rate: f64 = sample_rate * TWELFTH_ROOT_OF_TWO_POWERS[usize::from(best_note_name)];
    base_note += i16::from(best_note_name);
    // The last adjustment may have kicked us out of the desired range, so make
    // one final adjustment.
    if best_sample_rate < sample_rate_lower_bound {
        best_sample_rate *= 2.;
        base_note += 12;
    } else if best_sample_rate >= sample_rate_upper_bound {
        best_sample_rate /= 2.;
        base_note -= 12;
    }
    
    println!("3: {}", best_sample_rate);
    
    let best_sample_rate_int = best_sample_rate.round() as u32;
    println!("4: {} w/ base note {}", best_sample_rate_int, base_note);
    // Verify that this best sample rate + base note actually gives us back the
    // right pitch value
    let pitch_factor = (f64::from(60 - base_note)/12.).exp2() * 1024.;
    println!("Scaling sample rate up by {} ({})", pitch_factor, pitch_factor / 1024.);
    let calculated_sample_pitch = ((f64::from(best_sample_rate_int) * pitch_factor).round()) as u32;
    println!("Calculated sample pitch is {} (vs original {})", calculated_sample_pitch, sample_pitch);
    if calculated_sample_pitch == sample_pitch {
        return Some((best_sample_rate_int, base_note.try_into().unwrap()));
    }
    
    // If that didn't work, the caller will have to double the sample rate and
    // try again.
    // Or just assume a different sample rate... or something.
    None
}

fn convert_to_wav(gba_sample: &[u8]) -> Result<(Vec<u8>, u8), anyhow::Error> {
    let sample_size = u32::from_le_bytes(gba_sample[12..16].try_into().unwrap());
    let loop_start = u32::from_le_bytes(gba_sample[8..12].try_into().unwrap());
    let gba_pitch = u32::from_le_bytes(gba_sample[4..8].try_into().unwrap());
    let word0 = u32::from_le_bytes(gba_sample[0..4].try_into().unwrap());
    
    let sample_size_usize = usize::try_from(sample_size).unwrap();
    
    if gba_sample.len() < sample_size_usize || gba_sample.len() - sample_size_usize < 0x10 {
        bail!("Sample header length is longer than the actual file");
    }
    if (word0 & 0xFFFFFF) != 0 {
        bail!("Compressed samples are not supported");
    }
    let looping = (word0 & 0xC0000000) != 0;
    if looping && loop_start >= sample_size {
        bail!("Loop point is out of bounds");
    }
    
    let (sample_rate, base_note) = guess_base_note_from_sample_pitch(gba_pitch, 10000)
        .ok_or_else(|| anyhow!("Couldn't guess sample rate"))?;
    
    let allocation_size = 12usize + 8 + 16 + if looping { 8 + 15 * 4 } else { 0 } + 8 + sample_size_usize;
    let mut wav: Vec<u8> = Vec::with_capacity(allocation_size);
    // RIFF form magic
    wav.extend_from_slice(b"RIFF");
    // Form size
    wav.extend_from_slice(&u32::try_from(allocation_size - 8).unwrap().to_le_bytes());
    // Form type
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    // chunk ID
    wav.extend_from_slice(b"fmt ");
    // chunk size
    wav.extend_from_slice(&16u32.to_le_bytes());
    // wFormatTag -- PCM
    wav.extend_from_slice(&1u16.to_le_bytes());
    // wChannels -- mono
    wav.extend_from_slice(&1u16.to_le_bytes());
    // dwSamplesPerSec -- the sample rate
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    // dwAvgBytesPerSec -- sample rate * size in bytes of each sample/frame
    wav.extend_from_slice(&(sample_rate * 1).to_le_bytes());
    // wBlockAlign -- alignment of the wav data (does this cause padding at the end of the file...?)
    wav.extend_from_slice(&1u16.to_le_bytes());
    // wBitsPerSample -- GBA samples have a bit depth of 8
    wav.extend_from_slice(&8u16.to_le_bytes());
    
    if looping {
        // smpl chunk
        // chunk ID
        wav.extend_from_slice(b"smpl");
        // chunk size
        wav.extend_from_slice(&60u32.to_le_bytes());
        // dwManufacturer -- no specific manufacturer
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwProduct -- no specific product
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwSamplePeriod -- (sample rate)^-1 in nanoseconds, rounded down
        // ("For example, 44.1 kHz would be specified as 22675," not 22676)
        wav.extend_from_slice(&u32::try_from(1000000000u64 / u64::from(sample_rate)).unwrap().to_le_bytes());
        // dwMIDIUnityNote -- fill in a dummy value because I think no one cares about this
        wav.extend_from_slice(&60u32.to_le_bytes());
        // dwMIDIPitchFraction -- the sample rate guessing code assumes this is always 0
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwSMPTEFormat -- doesn't apply to us
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwSMPTEOffset -- doesn't apply to us
        wav.extend_from_slice(&0u32.to_le_bytes());
        // cSampleLoops -- we want to store one
        wav.extend_from_slice(&1u32.to_le_bytes());
        // cbSamplerData -- no extra data is necessary
        wav.extend_from_slice(&0u32.to_le_bytes());
        
        // sampler-loops
        // dwIdentifier -- name of the chunk. I think this is optional...?
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwType -- normal loop forward
        wav.extend_from_slice(&1u32.to_le_bytes());
        // dwStart -- loop start point
        wav.extend_from_slice(&loop_start.to_le_bytes());
        // dwEnd -- loop end point
        wav.extend_from_slice(&sample_size.to_le_bytes());
        // dwFraction -- no
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwPlayCount -- infinite loop
        wav.extend_from_slice(&0u32.to_le_bytes());
    }
    
    // data chunk
    // chunk ID
    wav.extend_from_slice(b"data");
    // chunk size
    wav.extend_from_slice(&sample_size.to_le_bytes());
    // wave data
    wav.extend_from_slice(&gba_sample[0x10usize..0x10usize+sample_size_usize]);
    let wav_len = wav.len();
    for i in 0..sample_size_usize {
        // Convert from unsigned 8-bit PCM to signed 8-bit PCM
        // by adding 0x80 to every value (wrapping)
        wav[wav_len - 1 - i] ^= 0x80;
    }
    
    Ok((wav, base_note))
}

fn main() -> Result<(), anyhow::Error> {
    // TODO: command-line argument parsing
    let in_path = PathBuf::from("../m3-732460.bin");
    let out_path = PathBuf::from("../m3-732460.wav");
    let error_if_outfile_exists = false;
    
    // Load all data from in_path
    let infile_data = std::fs::read(&in_path)
        .with_context(|| format!("Failed to read sample from {:?}", in_path))?;
    
    let (wav_data, base_note) = convert_to_wav(&infile_data[..])
        .with_context(|| format!("Failed to convert sample data from file {:?}", in_path))?;
    
    println!("Base note: {}{}", NOTE_NAMES[usize::from(base_note) % 12], base_note / 12 - 1);
    
    std::fs::write(&out_path, wav_data)
        .with_context(|| format!("Failed to write WAV to {:?}", out_path))?;
    
    Ok(())
}
