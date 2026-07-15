# Speech Commands

This example trains a small model from scratch to recognize eight closed-set
commands: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`. It does
not use pretrained weights or a TensorFlow runtime.

## Run The Workflow

```bash
nnctl data speech-commands
nnctl train speech-commands --output speech-commands.bin
nnctl run speech-commands -- \
  --model speech-commands.bin \
  --input data/mini_speech_commands/yes/example.wav
```

Pass `--input` more than once to rank several clips. Inference accepts PCM16
RIFF/WAV files with one or two channels. Clips are downmixed, resampled to
16 kHz, zero-padded to one second, and rejected if they are longer than one
second.

The frontend produces 24-band log-mel features pooled into 16 time bins. A
device-backed `[384, 64, 8]` MLP is trained with AdamW on speaker-disjoint
train, validation, and test splits. The `ZNSC` checkpoint stores the feature
configuration and normalization statistics as well as model weights.

This is a closed-set teaching demo. Audio that does not contain one of the
eight trained words will still be ranked as one of them; the probabilities are
not an unknown-word detector.

## Data Source And License

`nnctl data speech-commands` downloads the official
[Mini Speech Commands archive](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip)
used by TensorFlow's
[simple audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio).
The source Speech Commands dataset was collected by Google and released under
the Creative Commons Attribution 4.0 license. Prepared WAV data stays under
the ignored `data/` directory and is not redistributed by this repository.
