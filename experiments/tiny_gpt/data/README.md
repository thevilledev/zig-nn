# Tiny GPT Data Presets

The TinyGPT experiment is meant to grow through a small sequence of corpora.
Each preset keeps the same model code, but changes the amount and character of
language evidence the model sees.

## Presets

- `toy`: checked in at `experiments/tiny_gpt/data/toy.txt`. This is the offline
  smoke-test corpus used when no external data has been prepared.
- `shakespeare`: Tiny Shakespeare, the classic character-level dataset used by
  Karpathy's char-rnn and nanoGPT demos.
- `tinystories`: TinyStories, a synthetic short-story dataset designed for
  training and evaluating very small language models that still produce simple
  coherent English.

The executable defaults to `--corpus auto`, which uses `tinystories` if a
prepared TinyStories slice exists, then `shakespeare` if prepared, then `toy`.

## Prepare Data

Run:

```bash
bin/nnctl data tiny-gpt
```

This creates:

- `experiments/tiny_gpt/data/shakespeare/input.txt`
- `experiments/tiny_gpt/data/tinystories/valid.txt`
- `experiments/tiny_gpt/data/tinystories/tinystories_1mb.txt`

Generated corpora are intentionally ignored by git. The checked-in files only
describe how to reproduce them.

## Sources

- Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- nanoGPT Shakespeare prep reference: https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char
- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- TinyStories paper: https://arxiv.org/abs/2305.07759

TinyStories is listed on Hugging Face with the `cdla-sharing-1.0` license. Keep
that license in mind if you redistribute prepared slices.
