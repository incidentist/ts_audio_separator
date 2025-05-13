/**
 * TypeScript implementation of the Python STFT class from python-audio-separator
 * More closely follows the original structure and interface
 */

export class STFT {
  private fft_window: Float32Array;
  private hann_window: Float32Array;
  private n_fft: number;
  private hop_length: number;
  private dim_f: number;

  constructor(n_fft: number, hop_length: number, dim_f: number) {
    this.n_fft = n_fft;
    this.hop_length = hop_length;
    this.dim_f = dim_f;

    // Pre-compute windows
    this.fft_window = this.create_fft_window(n_fft);
    this.hann_window = this.create_hann_window(n_fft);
  }

  /**
   * Forward STFT - matches the Python implementation more closely
   * Input: [batch, channels, samples]
   * Output: [batch, channels * 2, dim_f, frames] where channels are doubled for real/imag
   */
  __call__(input: Float32Array[][]): Float32Array[][][][] {
    const batch_size = input.length;
    const num_channels = input[0].length;
    const signal_length = input[0][0].length;

    // Calculate number of frames
    const num_frames = Math.floor((signal_length - this.n_fft) / this.hop_length) + 1;

    // Initialize output array [batch, channels * 2, dim_f, frames]
    const output: Float32Array[][][][] = [];

    for (let b = 0; b < batch_size; b++) {
      const batch_output: Float32Array[][][] = [];

      // Process real and imaginary parts separately (like PyTorch)
      for (let part = 0; part < 2; part++) { // 0 = real, 1 = imag
        for (let ch = 0; ch < num_channels; ch++) {
          const channel_spec: Float32Array[][] = [];

          for (let f = 0; f < this.dim_f; f++) {
            channel_spec.push(new Float32Array(num_frames));
          }

          batch_output.push(channel_spec);
        }
      }

      output.push(batch_output);
    }

    // Process each batch
    for (let b = 0; b < batch_size; b++) {
      // Process each channel
      for (let ch = 0; ch < num_channels; ch++) {
        const signal = input[b][ch];

        // Process each frame
        for (let frame = 0; frame < num_frames; frame++) {
          const start = frame * this.hop_length;

          // Apply window
          const windowed = new Float32Array(this.n_fft);
          for (let i = 0; i < this.n_fft; i++) {
            if (start + i < signal.length) {
              windowed[i] = signal[start + i] * this.fft_window[i];
            }
          }

          // Perform FFT
          const fft_result = this.fft(windowed);

          // Store result (real parts in first half, imag parts in second half)
          for (let f = 0; f < this.dim_f && f < fft_result.length; f++) {
            output[b][ch][f][frame] = fft_result[f][0]; // Real
            output[b][ch + num_channels][f][frame] = fft_result[f][1]; // Imaginary
          }
        }
      }
    }

    return output;
  }

  /**
   * Inverse STFT - matches the Python implementation
   * Input: [batch, channels, dim_f, frames]
   * Output: [batch, channels/2, samples]
   */
  inverse(input: Float32Array[][][][]): Float32Array[][][] {
    const batch_size = input.length;
    const total_channels = input[0].length;
    const num_channels = total_channels / 2; // Real and imag parts
    const dim_f = input[0][0].length;
    const num_frames = input[0][0][0].length;

    // Calculate output length
    const output_length = (num_frames - 1) * this.hop_length + this.n_fft;

    // Initialize output [batch, channels, samples]
    const output: Float32Array[][][] = [];
    for (let b = 0; b < batch_size; b++) {
      const batch_output: Float32Array[][] = [];
      for (let ch = 0; ch < num_channels; ch++) {
        batch_output.push(new Float32Array(output_length));
      }
      output.push(batch_output);
    }

    // Window sum for normalization (like PyTorch)
    const window_sum: Float32Array[][] = [];
    for (let b = 0; b < batch_size; b++) {
      const batch_sum: Float32Array[] = [];
      for (let ch = 0; ch < num_channels; ch++) {
        batch_sum.push(new Float32Array(output_length));
      }
      window_sum.push(batch_sum);
    }

    // Process each batch
    for (let b = 0; b < batch_size; b++) {
      // Process each channel
      for (let ch = 0; ch < num_channels; ch++) {
        // Process each frame
        for (let frame = 0; frame < num_frames; frame++) {
          // Prepare complex spectrum
          const spectrum: Array<[number, number]> = [];

          // Get real and imaginary parts
          for (let f = 0; f < dim_f; f++) {
            const real = input[b][ch][f][frame];
            const imag = input[b][ch + num_channels][f][frame];
            spectrum.push([real, imag]);
          }

          // Mirror for negative frequencies (if needed)
          for (let f = dim_f; f < this.n_fft / 2 + 1; f++) {
            spectrum.push([0, 0]);
          }

          // Add conjugate for negative frequencies
          for (let f = 1; f < this.n_fft / 2; f++) {
            const idx = this.n_fft / 2 - f;
            if (idx < spectrum.length) {
              spectrum.push([spectrum[idx][0], -spectrum[idx][1]]);
            }
          }

          // Perform inverse FFT
          const time_signal = this.ifft(spectrum);

          // Apply window and overlap-add
          const start = frame * this.hop_length;
          for (let i = 0; i < this.n_fft && i < time_signal.length; i++) {
            if (start + i < output_length) {
              const windowed_sample = time_signal[i] * this.fft_window[i];
              output[b][ch][start + i] += windowed_sample;
              window_sum[b][ch][start + i] += this.fft_window[i] * this.fft_window[i];
            }
          }
        }

        // Normalize by window sum
        for (let i = 0; i < output_length; i++) {
          if (window_sum[b][ch][i] > 1e-8) {
            output[b][ch][i] /= window_sum[b][ch][i];
          }
        }
      }
    }

    // Trim to match input length (remove padding)
    const trimmed_output: Float32Array[][][] = [];
    for (let b = 0; b < batch_size; b++) {
      const batch_trimmed: Float32Array[][] = [];
      for (let ch = 0; ch < num_channels; ch++) {
        // Calculate expected length based on input
        const expected_length = (num_frames - 1) * this.hop_length + this.hop_length;
        const trimmed = output[b][ch].slice(0, Math.min(expected_length, output[b][ch].length));
        batch_trimmed.push(trimmed);
      }
      trimmed_output.push(batch_trimmed);
    }

    return trimmed_output;
  }

  private create_fft_window(n_fft: number): Float32Array {
    // Hann window for now (can be made configurable)
    return this.create_hann_window(n_fft);
  }

  private create_hann_window(n_fft: number): Float32Array {
    const window = new Float32Array(n_fft);
    for (let i = 0; i < n_fft; i++) {
      window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n_fft - 1));
    }
    return window;
  }

  // Use the existing FFT implementation
  private fft(input: Float32Array): Array<[number, number]> {
    // ... (existing FFT implementation)
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

  private bitReverse(x: number, n: number): number {
    let result = 0;
    let bits = Math.log2(n);

    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (x & 1);
      x >>= 1;
    }

    return result;
  }

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
