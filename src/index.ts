export { CommonSeparator } from './separator/common_separator';
export { MDXSeparator } from './separator/architectures/mdx_separator';
export { AudioUtils } from './utils/audio_utils';
export { STFT } from './separator/architectures/stft';
export type { SeparatorConfig } from './separator/common_separator';
export type { MDXArchConfig } from './separator/architectures/mdx_separator';

import * as tf from '@tensorflow/tfjs';

// Optional: explicitly set backend
await tf.setBackend('webgl'); // or 'webgpu' if available