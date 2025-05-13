# Web Demix2

A TypeScript library for running MDX audio demixing models in the browser using ONNX WebAssembly runtime.

## Structure

This project is structured similarly to the python-audio-separator repository, but implemented in TypeScript for browser usage.

- `src/separator/common_separator.ts` - Base class for all separator architectures
- `src/separator/architectures/mdx_separator.ts` - MDX architecture implementation
- `src/utils/audio_utils.ts` - Audio file loading and saving utilities
- `tests/` - Playwright tests
- `tests/fixtures/` - Test audio files location

## Test Audio Files

Place your sample WAV files in the `tests/fixtures/` directory. The test will automatically create a test WAV file, but you can also add your own audio files for testing.

## Setup

```bash
npm install
```

## Running Tests

```bash
npm test
```

## Architecture

The library follows the same structure as python-audio-separator:
- CommonSeparator base class
- MDXSeparator for MDX architecture models
- Methods use camelCase convention (e.g., `prepareMix` instead of `prepare_mix`)

## Current Implementation

Currently implemented:
- Project skeleton with TypeScript, Vue 3, and Playwright
- CommonSeparator base class with normalize and prepareMix methods
- MDXSeparator class with:
  - Model downloading from Hugging Face
  - prepareMix method
  - normalize method
  - initializeMix method for chunking audio
  - Basic demix method structure
  - Audio file loading via AudioUtils
- AudioUtils for loading/saving audio files
- Basic Playwright test setup with actual audio file testing
- Example HTML page demonstrating usage

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

## TODO

- Complete demix() method implementation
- Add STFT processing
- Implement model inference with ONNX runtime
- Complete the separation pipeline
- Add support for saving output files
- Add progress callbacks
- Implement match mix mode
- Add denoising support
