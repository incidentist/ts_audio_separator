/**
 * Adapter to make the Python-style STFT work with the MDX separator
 * This bridges between the batch-oriented Python interface and the simpler MDX usage
 */

import { STFT as PythonSTFT } from './stft-python-style';

export class STFT {
  private pythonStft: PythonSTFT;
  private n_fft: number;
  private hop_length: number;
  private dim_f: number;

  constructor(n_fft: number, hop_length: number, dim_f: number) {
    this.pythonStft = new PythonSTFT(n_fft, hop_length, dim_f);
    this.n_fft = n_fft;
    this.hop_length = hop_length;
    this.dim_f = dim_f;
  }

  /**
   * Forward STFT for MDX usage
   * Input: [channels, samples]
   * Output: [channels, dim_f, frames, 2] where last dimension is [real, imag]
   */
  forward(signal: Float32Array[], targetFrames?: number): Float32Array[][][][] {
    const channels = signal.length;
    let samples = signal[0].length;

    // Add padding if target frames specified
    if (targetFrames) {
      const requiredSamples = (targetFrames - 1) * this.hop_length + this.n_fft;
      if (samples < requiredSamples) {
        const paddedSignal: Float32Array[] = [];
        for (let ch = 0; ch < channels; ch++) {
          const padded = new Float32Array(requiredSamples);
          padded.set(signal[ch]);
          paddedSignal.push(padded);
        }
        signal = paddedSignal;
        samples = requiredSamples;
      }
    }

    // Convert to batch format [1, channels, samples]
    const batchInput: Float32Array[][][] = [[...signal]];

    // Call Python-style STFT [batch, channels * 2, dim_f, frames]
    const batchOutput = this.pythonStft.__call__(batchInput);

    // Convert from Python format to MDX format
    // Python: [batch=1, channels*2, dim_f, frames]
    // MDX: [channels, dim_f, frames, 2]
    const output: Float32Array[][][][] = [];

    for (let ch = 0; ch < channels; ch++) {
      const channelSpec: Float32Array[][][] = [];

      for (let f = 0; f < this.dim_f; f++) {
        const freqBin: Float32Array[][] = [];
        const numFrames = batchOutput[0][0][0].length;

        for (let frame = 0; frame < numFrames; frame++) {
          const real = batchOutput[0][ch][f][frame]; // Real part
          const imag = batchOutput[0][ch + channels][f][frame]; // Imaginary part
          freqBin.push(new Float32Array([real, imag]));
        }

        channelSpec.push(freqBin);
      }

      output.push(channelSpec);
    }

    return output;
  }

  /**
   * Inverse STFT for MDX usage
   * Input: [channels, dim_f, frames, 2] or number[][][][]
   * Output: [channels, samples]
   */
  inverse(spectrogram: Float32Array[][][][] | number[][][][]): Float32Array[] {
    const channels = spectrogram.length;
    const dim_f = spectrogram[0].length;
    const numFrames = spectrogram[0][0].length;

    // Convert from MDX format to Python format
    // MDX: [channels, dim_f, frames, 2]
    // Python: [batch=1, channels*2, dim_f, frames]
    const batchInput: Float32Array[][][][] = [];
    const batchChannels: Float32Array[][][] = [];

    // First add real parts for all channels
    for (let ch = 0; ch < channels; ch++) {
      const realChannel: Float32Array[][] = [];
      for (let f = 0; f < dim_f; f++) {
        const freqBin = new Float32Array(numFrames);
        for (let frame = 0; frame < numFrames; frame++) {
          freqBin[frame] = spectrogram[ch][f][frame][0]; // Real part
        }
        realChannel.push(freqBin);
      }
      batchChannels.push(realChannel);
    }

    // Then add imaginary parts for all channels
    for (let ch = 0; ch < channels; ch++) {
      const imagChannel: Float32Array[][] = [];
      for (let f = 0; f < dim_f; f++) {
        const freqBin = new Float32Array(numFrames);
        for (let frame = 0; frame < numFrames; frame++) {
          freqBin[frame] = spectrogram[ch][f][frame][1]; // Imaginary part
        }
        imagChannel.push(freqBin);
      }
      batchChannels.push(imagChannel);
    }

    batchInput.push(batchChannels);

    // Call Python-style inverse STFT
    const batchOutput = this.pythonStft.inverse(batchInput);

    // Extract from batch format [1, channels, samples] to [channels, samples]
    return batchOutput[0];
  }
}
