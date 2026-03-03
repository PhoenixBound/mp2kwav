use anyhow::{anyhow, bail, ensure};
use std::sync::LazyLock;

static TWELFTH_ROOT_OF_TWO_POWERS: LazyLock<[f64; 12]> = LazyLock::new(|| {
    [
        1.,                (1f64/12.).exp2(), (2f64/12.).exp2(),         (3f64/12.).exp2(),
        (4f64/12.).exp2(), (5f64/12.).exp2(), core::f64::consts::SQRT_2, (7f64/12.).exp2(),
        (8f64/12.).exp2(), (9f64/12.).exp2(), (10f64/12.).exp2(),        (11f64/12.).exp2()
    ]
});

fn calculate_sample_pitch(sample_rate: u32, base_note: i16) -> u32 {
    let sample_rate_float = f64::from(sample_rate);
    let pitch_factor = (f64::from(60 - base_note)/12.).exp2() * 1024.;
    let calculated_sample_pitch = ((sample_rate_float * pitch_factor).round()) as u32;
    calculated_sample_pitch
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
    // (NOTE: libm::remainder isn't a %. This is an IEEE remainder operation.
    //  It returns the difference from the nearest multiple of the divisor.
    //  As of writing, this operation isn't exposed directly in Rust, as far as I know.)
    let mut best_note_name = 0u8;
    let mut min_difference = libm::remainder(sample_rate, 1.0).abs();
    for i in 1u8..12 {
        let this_sample_rate = sample_rate * TWELFTH_ROOT_OF_TWO_POWERS[usize::from(i)];
        let score = libm::remainder(this_sample_rate, 1.0).abs();
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
        
    let best_sample_rate_int = best_sample_rate.round() as u32;
    // Verify that this best sample rate + base note actually gives us back the
    // right pitch value
    let calculated_sample_pitch = calculate_sample_pitch(best_sample_rate_int, base_note);
    if calculated_sample_pitch == sample_pitch {
        return Some((best_sample_rate_int, base_note.try_into().unwrap()));
    }
    
    // If that didn't work, the caller will have to double the sample rate and
    // try again.
    // Or just assume a different sample rate... or something.
    None
}

pub fn convert_to_wav(gba_sample: &[u8], min_rate: u32) -> Result<(Vec<u8>, u8), anyhow::Error> {
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
    
    let (sample_rate, base_note) = guess_base_note_from_sample_pitch(gba_pitch, min_rate)
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
        wav.extend_from_slice(&0u32.to_le_bytes());
        // dwStart -- loop start point
        wav.extend_from_slice(&loop_start.to_le_bytes());
        // dwEnd -- loop end point (inclusive in WAV files!)
        wav.extend_from_slice(&(sample_size - 1).to_le_bytes());
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

pub fn convert_to_gba(wav_sample: &[u8], base_note: u8) -> Result<Vec<u8>, anyhow::Error> {
    ensure!(wav_sample.len() >= 12 + 8 + 16 + 8, "WAV file isn't long enough to read metadata properly");
    ensure!(&wav_sample[0..4] == b"RIFF", "File is not a WAV file (bad RIFF form magic)");
    ensure!(u32::from_le_bytes(wav_sample[4..8].try_into().unwrap()).wrapping_add(8) == u32::try_from(wav_sample.len())?, "WAV size is wrong");
    ensure!(&wav_sample[8..12] == b"WAVE", "File is not a WAV file (other type of RIFF form)");
    
    let mut sample_rate: Option<u32> = None;
    let mut loop_points: Option<(u32, u32)> = None;
    let mut audio_data: Option<Vec<u8>> = None;
    let mut cursor = 12usize;
    while cursor < wav_sample.len() {
        if wav_sample.len() - cursor < 8 {
            bail!("Not enough data left in WAV file for chunk metadata");
        }
        let chunk_magic = &wav_sample[cursor..cursor+4];
        let chunk_size = u32::from_le_bytes(wav_sample[cursor+4..cursor+8].try_into().unwrap());
        let chunk_size_usize = usize::try_from(chunk_size).unwrap();
        if wav_sample.len() - (cursor + 8) < chunk_size_usize {
            bail!("Not enough data left in WAV file for {:?} chunk data", chunk_magic);
        }
        let chunk = &wav_sample[cursor+8..cursor+8+chunk_size_usize];
        
        // Handle stuff from known chunks
        match chunk_magic {
            b"fmt " => {
                ensure!(sample_rate.is_none(), "Multiple fmt chunks in .wav file");
                ensure!(chunk_size == 0x10, "fmt chunk in .wav file has length {} (expected 16; extended WAV headers aren't supported)", chunk_size);
                let format_tag = u16::from_le_bytes(chunk[0..2].try_into().unwrap());
                ensure!(format_tag == 1, "WAV file does not contain uncompressed PCM audio (expected: 1; actual: {}", format_tag);
                let channels = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
                ensure!(channels == 1, "WAV file has {} channels of audio, expected 1. Make it mono please.", channels);
                let samples_per_sec = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let avg_bytes_per_sec = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let block_align = u16::from_le_bytes(chunk[12..14].try_into().unwrap());
                let bits_per_sample = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
                ensure!(bits_per_sample == 8, "Unsupported bit depth {}. This tool only supports 8-bit WAVs.", bits_per_sample);
                
                let bytes_per_sample: u16 = (bits_per_sample + 7) / 8;
                let channels_u32 = u32::from(channels);
                ensure!(u32::from(bytes_per_sample) * samples_per_sec * channels_u32 == avg_bytes_per_sec, "WAV metadata is inconsistent");
                ensure!(channels * bytes_per_sample == block_align, "WAV alignment metadata is inconsistent");
                
                sample_rate = Some(samples_per_sec);
            }
            b"smpl" => {
                ensure!(loop_points.is_none(), "Multiple smpl chunks in .wav file");
                ensure!(chunk_size >= 36, "smpl chunk isn't long enough to read metadata (expected length: 36+; actual length: {}", chunk_size);
                // We don't really care about most of these fields.
                // Verifying sample_period is also probably more error-prone than verifying other
                // fields, since it's designed to make differing from the nominal sample rate
                // possible...
                
                // let manufacturer = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                // let product = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                // let sample_period = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                // let midi_unity_note = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
                // let midi_pitch_fraction = u32::from_le_bytes(chunk[16..20].try_into().unwrap());
                // let smpte_format = u32::from_le_bytes(chunk[20..24].try_into().unwrap());
                // let smpte_offset = u32::from_le_bytes(chunk[24..28].try_into().unwrap());
                let sample_loops_count = u32::from_le_bytes(chunk[28..32].try_into().unwrap());
                ensure!(sample_loops_count >= 1, "No sample loops in smpl chunk. Why is it even here?");
                let sampler_data_byte_count = u32::from_le_bytes(chunk[32..36].try_into().unwrap());
                ensure!(chunk_size == 36 + sample_loops_count * 24 + sampler_data_byte_count, "smpl chunk size metadata is inconsistent");
                
                // Read the first loop in the sampler-loops data and hope that it's the right one.
                // let identifier = u32::from_le_bytes(chunk[36..40].try_into().unwrap());
                let type_ = u32::from_le_bytes(chunk[40..44].try_into().unwrap());
                ensure!(type_ == 0, "Sampler loop uses weird loop type");
                let loop_start = u32::from_le_bytes(chunk[44..48].try_into().unwrap());
                let loop_end = u32::from_le_bytes(chunk[48..52].try_into().unwrap());
                let fraction = u32::from_le_bytes(chunk[52..56].try_into().unwrap());
                // let play_count = u32::from_le_bytes(chunk[56..60].try_into().unwrap());
                ensure!(loop_start <= loop_end, "Loop starting point exceeds loop ending point");
                ensure!(fraction == 0, "Sampler loop involves fractional positions");
                if let Some(ref mut audio_data_unwrapped) = audio_data {
                    let loop_end_usize = usize::try_from(loop_end).unwrap();
                    ensure!(loop_end_usize < audio_data_unwrapped.len(), "Loop end point lies outside of sample");
                    for _ in loop_end_usize..audio_data_unwrapped.len() - 1 {
                        _ = audio_data_unwrapped.pop();
                    }
                }
                
                loop_points = Some((loop_start, loop_end));
            }
            b"data" => {
                ensure!(sample_rate.is_some(), "No fmt chunk before data chunk");
                ensure!(audio_data.is_none(), "Multiple data chunks in .wav file");
                
                let mut audio_data_limit = chunk_size_usize;
                if let Some((_, loop_end)) = loop_points {
                    ensure!(loop_end < chunk_size, "Loop end point lies outside of sample");
                    audio_data_limit = usize::try_from(loop_end).unwrap() + 1;
                }
                
                let mut audio_data_vec = Vec::<u8>::with_capacity(audio_data_limit);
                audio_data_vec.extend_from_slice(&chunk[0..audio_data_limit]);
                for i in 0..audio_data_limit {
                    // Convert unsigned 8-bit PCM to signed 8-bit PCM by subtracting 0x80
                    audio_data_vec[i] ^= 0x80;
                }
                audio_data = Some(audio_data_vec);
            }
            _ => {}
        }
        
        cursor += 8;
        cursor += chunk_size_usize;
    }
    
    ensure!(sample_rate.is_some(), "No fmt chunk in .wav file");
    ensure!(audio_data.is_some(), "No data chunk in .wav file");
    
    // Now... we write the GBA sample data
    let audio_data = audio_data.unwrap();
    let mut output = Vec::<u8>::with_capacity(audio_data.len() + 0x11);
    output.extend_from_slice(b"\x00\x00\x00\x00");
    output.extend_from_slice(&calculate_sample_pitch(sample_rate.unwrap(), base_note.into()).to_le_bytes()[..]);
    if let Some((loop_start, _)) = loop_points {
        output[3] = 0x40;
        output.extend_from_slice(&loop_start.to_le_bytes()[..]);
    } else {
        output.extend_from_slice(b"\x00\x00\x00\x00");
    }
    output.extend_from_slice(&u32::try_from(audio_data.len()).unwrap().to_le_bytes()[..]);
    output.extend_from_slice(&audio_data[..]);
    output.push(if let Some((loop_start, _)) = loop_points {
        audio_data[usize::try_from(loop_start).unwrap()]
    } else {
        0
    });
    Ok(output)
}
