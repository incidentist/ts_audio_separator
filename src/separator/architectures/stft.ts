import * as tf from '@tensorflow/tfjs';
import { Logger } from '../common_separator';

export class STFT {
  /**
   * This class performs the Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
   * These functions are essential for converting the audio between the time domain and the frequency domain,
   * which is a crucial aspect of audio processing in neural networks.
   */
  private logger: Logger;
  private n_fft: number;
  private hop_length: number;
  private dim_f: number;
  private hann_window: tf.Tensor;

  constructor(logger: Logger, n_fft: number, hop_length: number, dim_f: number) {
    this.logger = logger;
    this.n_fft = n_fft;
    this.hop_length = hop_length;
    this.dim_f = dim_f;

    // Create a Hann window tensor for use in the STFT.
    this.hann_window = tf.signal.hannWindow(this.n_fft);

    this.logger.debug(`STFT initialized: n_fft=${n_fft}, hop_length=${hop_length}, dim_f=${dim_f}`);
  }

  forward(input_tensor: Float32Array[][]): number[][][][] {
    // Convert input to tensor
    let tensor = tf.tensor3d(input_tensor);

    // Extract batch dimensions (all dimensions except the last two which are channel and time).
    const batchShape = tensor.shape.slice(0, -2);
    const batchDimensions = batchShape.length > 0 ? batchShape : [1];

    // Extract channel and time dimensions (last two dimensions of the tensor).
    const channel_dim = tensor.shape[tensor.shape.length - 2];
    const time_dim = tensor.shape[tensor.shape.length - 1];

    // Reshape the tensor to merge batch and channel dimensions for STFT processing.
    const reshaped_tensor = tensor.reshape([-1, time_dim]);

    // Transfer the pre-defined window tensor to the same device as the input tensor.
    const stft_window = this.hann_window;

    // Perform the Short-Time Fourier Transform (STFT) on the reshaped tensor.
    // Note: TensorFlow.js stft returns complex tensor directly
    const stft_output = tf.signal.stft(
      reshaped_tensor,
      this.n_fft,
      this.hop_length,
      this.n_fft,
      () => stft_window
    );

    // Convert complex to real and imaginary parts
    const real = tf.real(stft_output);
    const imag = tf.imag(stft_output);

    // Stack real and imaginary parts along a new dimension
    const stacked = tf.stack([real, imag], 3);

    // Rearrange the dimensions to match PyTorch output format
    // PyTorch: [batch*channel, freq, time, 2] -> permute to [batch*channel, 2, freq, time]
    const permuted_stft_output = stacked.transpose([0, 3, 1, 2]);

    // Reshape the output to restore the original batch and channel dimensions
    const output_shape = [...batchDimensions, channel_dim, 2, permuted_stft_output.shape[2], permuted_stft_output.shape[3]];
    let final_output = permuted_stft_output.reshape(output_shape);

    // Merge the complex dimension with channel dimension
    const merged_shape = [...batchDimensions, channel_dim * 2, final_output.shape[final_output.shape.length - 2], final_output.shape[final_output.shape.length - 1]];
    final_output = final_output.reshape(merged_shape);

    // Return the transformed tensor, sliced to retain only the required frequency dimension (`dim_f`).
    const sliced = final_output.slice(
      Array(final_output.shape.length - 2).fill(0).concat([0, 0]),
      Array(final_output.shape.length - 2).fill(-1).concat([this.dim_f, -1])
    );

    const result = sliced.arraySync() as number[][][][];

    // Clean up tensors
    tensor.dispose();
    reshaped_tensor.dispose();
    stft_output.dispose();
    real.dispose();
    imag.dispose();
    stacked.dispose();
    permuted_stft_output.dispose();
    final_output.dispose();
    sliced.dispose();

    return result;
  }

  private pad_frequency_dimension(
    input_tensor: tf.Tensor,
    batch_dimensions: number[],
    channel_dim: number,
    freq_dim: number,
    time_dim: number,
    num_freq_bins: number
  ): tf.Tensor {
    /**
     * Adds zero padding to the frequency dimension of the input tensor.
     */
    // Create a padding tensor for the frequency dimension
    const padding_shape = [...batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim];
    const freq_padding = tf.zeros(padding_shape);

    // Concatenate the padding to the input tensor along the frequency dimension.
    const padded_tensor = tf.concat([input_tensor, freq_padding], input_tensor.shape.length - 2);

    freq_padding.dispose();

    return padded_tensor;
  }

  private calculate_inverse_dimensions(input_tensor: tf.Tensor): [number[], number, number, number, number] {
    // Extract batch dimensions and frequency-time dimensions.
    const batch_dimensions = input_tensor.shape.slice(0, -3);
    const channel_dim = input_tensor.shape[input_tensor.shape.length - 3];
    const freq_dim = input_tensor.shape[input_tensor.shape.length - 2];
    const time_dim = input_tensor.shape[input_tensor.shape.length - 1];

    // Calculate the number of frequency bins for the inverse STFT.
    const num_freq_bins = Math.floor(this.n_fft / 2) + 1;

    return [batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins];
  }

  private prepare_for_istft(
    padded_tensor: tf.Tensor,
    batch_dimensions: number[],
    channel_dim: number,
    num_freq_bins: number,
    time_dim: number
  ): tf.Tensor {
    /**
     * Prepares the tensor for Inverse Short-Time Fourier Transform (ISTFT) by reshaping
     * and creating a complex tensor from the real and imaginary parts.
     */
    // Reshape the tensor to separate real and imaginary parts and prepare for ISTFT.
    const reshaped_shape = [...batch_dimensions, Math.floor(channel_dim / 2), 2, num_freq_bins, time_dim];
    const reshaped_tensor = padded_tensor.reshape(reshaped_shape);

    // Flatten batch dimensions and rearrange for ISTFT.
    const flattened_tensor = reshaped_tensor.reshape([-1, 2, num_freq_bins, time_dim]);

    // Rearrange the dimensions of the tensor to bring the frequency dimension forward.
    const permuted_tensor = flattened_tensor.transpose([0, 2, 3, 1]);

    // Extract real and imaginary parts
    const real = permuted_tensor.slice(
      [0, 0, 0, 0],
      [-1, -1, -1, 1]
    ).squeeze([3]);

    const imag = permuted_tensor.slice(
      [0, 0, 0, 1],
      [-1, -1, -1, 1]
    ).squeeze([3]);

    // Combine real and imaginary parts into a complex tensor.
    const complex_tensor = tf.complex(real, imag);

    // Clean up intermediate tensors
    reshaped_tensor.dispose();
    flattened_tensor.dispose();
    permuted_tensor.dispose();
    real.dispose();
    imag.dispose();

    return complex_tensor;
  }

  inverse(stft_matrix: number[][][][]): Float32Array[][] {
    // Convert input to tensor
    let input_tensor = tf.tensor4d(stft_matrix);

    // Transfer the pre-defined Hann window tensor to the same device as the input tensor.
    const stft_window = this.hann_window;

    const [batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins] = this.calculate_inverse_dimensions(input_tensor);

    const padded_tensor = this.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins);

    const complex_tensor = this.prepare_for_istft(padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim);

    // Perform the Inverse Short-Time Fourier Transform (ISTFT).
    // Note: TensorFlow.js doesn't have istft, so we need to use irfft and implement overlap-add
    const time_signal = tf.signal.irfft(complex_tensor);

    // Apply overlap-add reconstruction
    const num_frames = time_signal.shape[1];
    const frame_length = time_signal.shape[2];
    const num_samples = (num_frames - 1) * this.hop_length + frame_length;

    // Initialize output buffer
    const batch_channel_size = time_signal.shape[0];
    const outputBuffer = tf.buffer([batch_channel_size, num_samples]);
    const windowArray = stft_window.arraySync() as number[];
    const normBuffer = tf.buffer([batch_channel_size, num_samples]);

    const timeSignalArray = time_signal.arraySync() as number[][][];

    // Perform overlap-add
    for (let bc = 0; bc < batch_channel_size; bc++) {
      for (let t = 0; t < num_frames; t++) {
        const offset = t * this.hop_length;
        for (let i = 0; i < frame_length && offset + i < num_samples; i++) {
          const value = timeSignalArray[bc][t][i] * windowArray[i];
          outputBuffer.set(outputBuffer.get(bc, offset + i) + value, bc, offset + i);
          normBuffer.set(normBuffer.get(bc, offset + i) + windowArray[i] * windowArray[i], bc, offset + i);
        }
      }
    }

    // Normalize
    const outputArray = outputBuffer.toTensor();
    const normArray = normBuffer.toTensor();
    const normalized = outputArray.div(normArray.add(1e-8));

    // Reshape ISTFT result to restore original batch and channel dimensions.
    const output_shape = [...batch_dimensions, 2, -1];
    const final_output = normalized.reshape(output_shape);

    const result = final_output.arraySync() as Float32Array[][];

    // Clean up tensors
    input_tensor.dispose();
    padded_tensor.dispose();
    complex_tensor.dispose();
    time_signal.dispose();
    outputArray.dispose();
    normArray.dispose();
    normalized.dispose();
    final_output.dispose();

    return result;
  }
}
