# mp2kwav

A sample converter which converts sample data to and from .wav, targeting the
SDK/BIOS sound engine on the GBA (MP2K). Unlike other sample converters, it
deals with samples that have already been extracted into separate files from the
original ROM.

I wrote this because I was dissatisfied with a similar tool that worked with
.aif files. The use of AIFF allowed that tool, aif2pcm, to be lazy and not deal
with important usability issues:

* **You can't specify the "root note" or "base note" of a sample in aif2pcm.**
  This feature isn't built into the sound engine's instrument data itself;
  instead, the sample rate of the sample is given 10 fractional bits, so that
  the sample rate of middle C (the fixed base note) can be specified at
  relatively high precision even if the sample was for a different note. (Shout
  out to Taihennami from the Musicombo sorting algorithm Discord server, for
  pointing out that 10 fractional bits is enough to give a precision of roughly
  ±1 mHz (millihertz)).
* Related to the above, samples in GBA games will often have non-integer sample
  rates, and **these non-integer sample rates can be copied wholesale into AIFF
  files only**. RIFF WAVE and most other file formats I know store the sample
  rate as an integer. This makes it difficult to convert away from AIFF without
  losing tuning information.
* No modern audio editing tools support editing loops in AIFF files. (I asked
  one Pokémon ROM hacker how to do that, and his response (paraphrased) was "I
  convert it to GBA format, hex edit in the loop point, and convert back." Not
  ideal...)

This tool attempts to deal with all of those issues:

* You can specify the root note of a sample. The tool will try to guess the
  lowest possible base note given a user-specified lower bound on the sample
  rate. This makes it easier to predict how instruments that use multiple
  samples actually sound from a source checkout.
* Samples will have sane integer sample rates. The risk of losing precision when
  saving an edited sample is greatly diminished.
* Many tools exist that can edit "smpl" chunks in .wav files to change loop
  start and end points. It's a pretty popular mechanism for making looping
  audio.

## To-do list

That said, this tool is *itself* a lazy minimum viable product with messy code.
It's missing many features that are probably important for some use cases:

* Support for directly taking in 16-bit .wav files as input. aif2pcm supports
  this.
* Support for compressed samples like those used in the Pokémon variant of the
  sound engine.
* Support for base notes that are offset by some number of cents. (I thought
  this feature was useless until I heard that some SNES sample designers take
  advantage of exact sample rates and frequencies before ADPCM conversion to
  [maximize the chance of a clean loop point](https://nesdoug.com/2022/01/27/why-b21-cents/).
  Who knows if any GBA games actually did anything like that, though?)
* Support for overriding the base note guessing procedure used by this tool.
  (This will probably accompany support for the above cents offset feature.)

## Build instructions

To build the tool, [install the Rust compiler and Cargo package manager](https://www.rust-lang.org/learn/get-started)
via Rustup or your package manager, and run `cargo build --release`. The
compiled program can be found in the target/release folder.

I've tested building everything with Rust 1.83.0 on Windows 10 x64. Everything
should definitely work on that platform (and most likely other platforms too).

## Usage instructions

To convert a sample from GBA format (.bin) to .wav:

```
mp2kwav [-m<number>|--min-rate=<number>] <gba-sample.bin> <output-wav-sample.wav>
```

To convert a sample from .wav into the GBA format:

```
mp2kwav [-b<note>|--base=<note>] <wav-sample.wav> <output-gba-sample.bin>
```

(`note` is spelled out as in [American Standard Pitch Notation](https://en.wikipedia.org/wiki/Scientific_pitch_notation);
middle C is C4, and 440 Hz is A4. Sharps are spelled with `#` and flats with
`b`, for notes that require accidentals to write out.)

See the built-in help (`mp2kwav --help`) for more information.
