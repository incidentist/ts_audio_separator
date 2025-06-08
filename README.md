# Web Demix2

A TypeScript library for running MDX audio demixing models in the browser using ONNX WebAssembly runtime.

This is a TypeScript translation of [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), starting with the `UVR_MDXNET_KARA_2` model. Most of the code is dedicated to converting the input audio into a format that the underlying ONNX model expects, and then converting the output of the model back to normal audio. This is complicated for the following reasons:

1. I (incidentist) don't know much about audio processing, so I'm relying on AI, for better or worse. I am in way over my head and it is a fun learning experience but I *do* want to get it actually working.
2. The python version uses pytorch for fast vector/matrix math. Pytorch doesn't run on TypeScript, so we are translating it into tensorflow.js, the closest equivalent that will run in a browser. tfjs lacks a lot of the functions that pytorch has, and some of its implementations differ from pytorch in important ways.

Install with `npm install`.

Test using:

npx playwright test tests/mdx-separator.spec.ts

This is also a good way to see the intended usage of the library.

## Current Status - 6/8/2025
1. The main integration test passes, and the resulting audio files indicate that the model is indeed separating audio. A 3.5 minute song takes 12 minutes to separate on an M1 Pro.
2. The resulting audio files are not correct. instrumental.wav has no vocals (yay!) but it doesn't have enough of the instrumentals. It sounds muted and a little distorted. The vocals.wav file, on the other hand, still has a lot of instrumentals (which makes sense because it is just the result of the full mix minus the model-generated instrumental file). So the main next step is to figure out which part(s) of the pre/post processing are causing the sub-optimal result.
3. A big part of this project is an stft library that a) mimics the pytorch implementation of stft, and b) includes an implementation of istft, which is not in tensorflow.js. `torchStyleStft()` produces correct output, and I believe the whole `forward()` function produces correct output. But `inverse(forward(t))` should be equal to `t` and that is not currently happening. `src/separator/architecture/stft.spec.ts` is the test for this. I suspect there are a few parts of inverse() that are not doing the right thing, because we had to write more of it from scratch. 
4. One thing I have not done is log info about tensors at each part of the Python process, and verify that the info matches up at each part of the TypeScript process. I think this will be very useful to see where they diverge.

## Structure

This project is structured similarly to the python-audio-separator repository, but implemented in TypeScript for browser usage. It uses tensorflow.js instead of PyTorch for vector/matrix manipulation.

- `src/separator/common_separator.ts` - Base class for all separator architectures
- `src/separator/architectures/mdx_separator.ts` - MDX architecture implementation
- `src/utils/audio_utils.ts` - Audio file loading and saving utilities
- `tests/` - Playwright tests
- `tests/fixtures/` - Test audio files location

## Test Audio Files

Place your sample WAV files in the `tests/fixtures/` directory. The main mdx-separator integration test uses a 1s audio file. The stft test will automatically create a test WAV file, but you can also add your own audio files for testing.

## Usage Example

```typescript
import { MDXSeparator } from 'web-demix2';

const separator = new MDXSeparator(
  {
    modelPath: 'UVR_MDXNET_KARA_2.onnx',
    modelName: 'UVR_MDXNET_KARA_2',
    modelData: {
      compensate: 1.0,
      mdx_dim_f_set: 3072,
      mdx_dim_t_set: 8,
      mdx_n_fft_scale_set: 7680
    }
  },
  {
    segmentSize: 256,
    overlap: 0.25,
    batchSize: 1,
    hopLength: 1024,
    enableDenoise: false
  }
);

// Load the model (downloads from Hugging Face)
await separator.loadModel();

// Process an audio file
const outputFiles = await separator.separate('path/to/audio.wav');
```

