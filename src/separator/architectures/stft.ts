/**
 * Short-Time Fourier Transform implementation for MDX separator
 * Provides forward and inverse STFT operations similar to PyTorch's STFT
 */

export class STFT {
  private nFft: number;
  private hopLength: number;
  private dimF: number;
  private window: Float32Array;

  constructor(nFft: number, hopLength: number, dimF: number) {
    this.nFft = nFft;
    this.hopLength = hopLength;
    this.dimF = dimF;

    // Create Hann window for STFT
    this.window = this.hannWindow(nFft);
  }

  /**
   * Perform forward STFT on input signal
   * @param signal Input audio signal [channels, samples]
   * @param targetFrames Optional target number of frames (for padding)
   * @returns Complex spectrogram [channels, dimF, frames, 2] where last dimension is [real, imag]
   */
  forward(signal: Float32Array[], targetFrames?: number): Float32Array[][][][] {
    const channels = signal.length;
    let samples = signal[0].length;
    let numFrames = Math.floor((samples - this.nFft) / this.hopLength) + 1;

    // If target frames specified and we have fewer, pad the signal
    if (targetFrames && numFrames < targetFrames) {
      const samplesNeeded = (targetFrames - 1) * this.hopLength + this.nFft;
      const paddingNeeded = samplesNeeded - samples;

      const paddedSignal: Float32Array[] = [];
      for (let ch = 0; ch < channels; ch++) {
        const padded = new Float32Array(samplesNeeded);
        padded.set(signal[ch]);
        paddedSignal.push(padded);
      }
      signal = paddedSignal;
      samples = samplesNeeded;
      numFrames = targetFrames;
    }

    // Initialize output array [channels, dimF, frames, 2]
    const spectrogram: Float32Array[][][][] = [];

    for (let ch = 0; ch < channels; ch++) {
      const channelSpec: Float32Array[][][] = [];

      for (let f = 0; f < this.dimF; f++) {
        const freqBin: Float32Array[][] = [];

        for (let frame = 0; frame < numFrames; frame++) {
          // Each frame contains [real, imag] parts
          freqBin.push(new Float32Array([0, 0]));
        }

        channelSpec.push(freqBin);
      }

      spectrogram.push(channelSpec);
    }

    // Process each channel
    for (let ch = 0; ch < channels; ch++) {
      const channelSignal = signal[ch];

      // Process each frame
      for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * this.hopLength;
        const end = start + this.nFft;

        // Extract frame and apply window
        const windowedFrame = new Float32Array(this.nFft);
        for (let i = 0; i < this.nFft; i++) {
          if (start + i < channelSignal.length) {
            windowedFrame[i] = channelSignal[start + i] * this.window[i];
          }
        }

        // Perform FFT on windowed frame
        const fftResult = this.fft(windowedFrame);

        // Copy relevant frequency bins to output
        for (let f = 0; f < this.dimF && f < fftResult.length; f++) {
          spectrogram[ch][f][frame][0] = fftResult[f][0]; // Real part
          spectrogram[ch][f][frame][1] = fftResult[f][1]; // Imaginary part
        }
      }
    }

    return spectrogram;
  }

  /**
   * Perform inverse STFT to reconstruct signal from spectrogram
   * @param spectrogram Complex spectrogram [channels, dimF, frames, 2]
   * @returns Reconstructed audio signal [channels, samples]
   */
  inverse(spectrogram: Float32Array[][][][] | number[][][][]): Float32Array[] {
    const channels = spectrogram.length;
    const numFrames = spectrogram[0][0].length;
    const outputLength = (numFrames - 1) * this.hopLength + this.nFft;

    // Initialize output signal
    const signal: Float32Array[] = [];
    for (let ch = 0; ch < channels; ch++) {
      signal.push(new Float32Array(outputLength));
    }

    // Window for overlap-add
    const window = this.window;

    // Process each channel
    for (let ch = 0; ch < channels; ch++) {
      // Process each frame
      for (let frame = 0; frame < numFrames; frame++) {
        // Prepare complex spectrum for this frame
        const frameSpectrum: Array<[number, number]> = [];

        // Copy positive frequencies
        for (let f = 0; f < this.dimF; f++) {
          const real = spectrogram[ch][f][frame][0];
          const imag = spectrogram[ch][f][frame][1];
          frameSpectrum.push([real, imag]);
        }

        // Mirror negative frequencies (excluding DC and Nyquist)
        for (let f = this.dimF - 2; f > 0; f--) {
          const real = spectrogram[ch][f][frame][0];
          const imag = -spectrogram[ch][f][frame][1]; // Complex conjugate
          frameSpectrum.push([real, imag]);
        }

        // Perform inverse FFT
        const timeDomain = this.ifft(frameSpectrum);

        // Apply window and overlap-add
        const start = frame * this.hopLength;
        for (let i = 0; i < this.nFft && i < timeDomain.length; i++) {
          if (start + i < outputLength) {
            signal[ch][start + i] += timeDomain[i] * window[i];
          }
        }
      }
    }

    return signal;
  }

  /**
   * Create Hann window
   */
  private hannWindow(size: number): Float32Array {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1));
    }
    return window;
  }

  /**
   * Fast Fourier Transform using Cooley-Tukey algorithm
   * Returns array of [real, imag] pairs
   */
  private fft(input: Float32Array): Array<[number, number]> {
    const n = input.length;
    const output: Array<[number, number]> = new Array(n);

    // Initialize with zeros
    for (let i = 0; i < n; i++) {
      output[i] = [0, 0];
    }

    // Base case
    if (n <= 1) {
      if (n === 1) {
        output[0] = [input[0], 0];
      }
      return output;
    }

    // Make sure n is a power of 2 by padding if necessary
    let paddedSize = 1;
    while (paddedSize < n) paddedSize *= 2;

    if (n !== paddedSize) {
      // Pad with zeros
      const paddedInput = new Float32Array(paddedSize);
      paddedInput.set(input);
      return this.fftRadix2(paddedInput).slice(0, n);
    }

    return this.fftRadix2(input);
  }

  /**
   * Radix-2 FFT implementation
   */
  private fftRadix2(input: Float32Array): Array<[number, number]> {
    const n = input.length;

    if (n <= 1) {
      return [[input[0] || 0, 0]];
    }

    // Bit reversal
    const output: Array<[number, number]> = new Array(n);
    for (let i = 0; i < n; i++) {
      output[i] = [input[this.bitReverse(i, n)], 0];
    }

    // Cooley-Tukey FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angleStep = -2 * Math.PI / size;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const angle = angleStep * j;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);

          const a = output[i + j];
          const b = output[i + j + halfSize];

          const tReal = b[0] * cos - b[1] * sin;
          const tImag = b[0] * sin + b[1] * cos;

          output[i + j] = [a[0] + tReal, a[1] + tImag];
          output[i + j + halfSize] = [a[0] - tReal, a[1] - tImag];
        }
      }
    }

    return output;
  }

  /**
   * Bit reversal for FFT
   */
  private bitReverse(x: number, n: number): number {
    let result = 0;
    let bits = Math.log2(n);

    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (x & 1);
      x >>= 1;
    }

    return result;
  }

  /**
   * Inverse Fast Fourier Transform
   * Takes array of [real, imag] pairs and returns real signal
   */
  private ifft(input: Array<[number, number]>): Float32Array {
    const n = input.length;

    // Make sure n is a power of 2
    let paddedSize = 1;
    while (paddedSize < n) paddedSize *= 2;

    let paddedInput = input;
    if (n !== paddedSize) {
      paddedInput = new Array(paddedSize);
      for (let i = 0; i < n; i++) {
        paddedInput[i] = input[i];
      }
      for (let i = n; i < paddedSize; i++) {
        paddedInput[i] = [0, 0];
      }
    }

    // Conjugate the input
    const conjugated = paddedInput.map(([real, imag]) => [real, -imag] as [number, number]);

    // Apply FFT
    const fftResult = this.fftRadix2(new Float32Array(conjugated.map(c => c[0])));

    // Conjugate and scale the output
    const output = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      output[i] = fftResult[i][0] / paddedSize;
    }

    return output;
  }
}
