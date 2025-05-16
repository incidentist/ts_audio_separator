import * as tf from '@tensorflow/tfjs';

export class STFT {
  /**
   * This class performs the Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
   * These functions are essential for converting the audio between the time domain and the frequency domain,
   * which is a crucial aspect of audio processing in neural networks.
   */
  private nFft: number;
  private hopLength: number;
  private dimF: number;
  private hannWindow: tf.Tensor;

  constructor(n_fft: number, hop_length: number, dim_f: number) {
    this.nFft = n_fft;
    this.hopLength = hop_length;
    this.dimF = dim_f;

    // Create a periodic Hann window (equivalent to PyTorch's periodic=True)
    const periodicHannWindow = tf.tensor(
      Array.from({ length: n_fft }, (_, i) => 0.5 * (1 - Math.cos(2 * Math.PI * i / n_fft)))
    );
    this.hannWindow = periodicHannWindow;

    console.debug(`STFT initialized: n_fft=${n_fft}, hop_length=${hop_length}, dim_f=${dim_f}`);
  }

  /**
   * Python-style call interface for compatibility
   * In Python, an instance can be called like a function: stft(tensor)
   */
  call(tensor: tf.Tensor): tf.Tensor {
    console.debug('STFT being called with Python-style interface');
    return this.forward(tensor);
  }

  // input: [1, 2, length]
  forward(tensor: tf.Tensor): tf.Tensor {
    // Perform the Short-Time Fourier Transform (STFT)
    console.debug(`Input tensor shape: ${tensor.shape}`);

    // For a batched input [batch, channels, time], we need to process each batch+channel separately
    const batchSize = tensor.shape.length > 2 ? tensor.shape[0] : 1;
    const channelCount = tensor.shape.length > 2 ? tensor.shape[1] : tensor.shape[0];
    const timeLength = tensor.shape.length > 2 ? tensor.shape[2] : tensor.shape[1];

    console.debug(`Parsed dimensions: batch=${batchSize}, channels=${channelCount}, time=${timeLength}`);

    // Process each batch and channel separately
    const outputs = [];
    for (let b = 0; b < batchSize; b++) {
      const batchOutputs = [];
      for (let ch = 0; ch < channelCount; ch++) {
        // Extract this single channel as a TRUE 1D tensor
        let channelData;
        if (tensor.shape.length > 2) {
          // For batched input: [batch, channels, time]
          channelData = tensor.slice([b, ch, 0], [1, 1, -1]).reshape([timeLength]);
        } else {
          // For non-batched input: [channels, time]
          channelData = tensor.slice([ch, 0], [1, -1]).reshape([timeLength]);
        }

        console.debug(`Batch ${b}, Channel ${ch} data shape: ${channelData.shape}`);

        // Manual centering like PyTorch's center=True
        const paddingSize = Math.floor(this.nFft / 2);
        const paddedData = tf.pad(channelData, [[paddingSize, paddingSize]], 'reflect');
        console.debug(`Padded data shape: ${paddedData.shape}`);

        // Apply STFT - with the TRUE 1D tensor (no batch dimension)
        const stftOutput = tf.signal.stft(
          paddedData,
          this.nFft,
          this.hopLength,
          this.nFft,
          () => this.hannWindow
        );

        console.debug(`STFT output shape for batch ${b}, channel ${ch}: ${stftOutput.shape}`);

        // Convert complex output to real/imag pair
        const realPart = tf.real(stftOutput);
        const imagPart = tf.imag(stftOutput);

        // Stack real and imaginary parts as the last dimension
        // Result shape will be [freqBins, frames, 2]
        const complexData = tf.stack([realPart, imagPart], -1);

        // Save this channel's output
        batchOutputs.push(complexData);

        // Clean up
        channelData.dispose();
        paddedData.dispose();
        stftOutput.dispose();
        realPart.dispose();
        imagPart.dispose();
      }

      // Stack channels for this batch
      // From: list of [freqBins, frames, 2]
      // To: [channels, freqBins, frames, 2]
      const batchStacked = tf.stack(batchOutputs, 0);
      outputs.push(batchStacked);

      // Clean up
      batchOutputs.forEach(c => c.dispose());
    }

    // Stack all batches
    // From: list of [channels, freqBins, frames, 2]
    // To: [batch, channels, freqBins, frames, 2]
    let result;
    if (batchSize > 1) {
      result = tf.stack(outputs, 0);
    } else {
      result = outputs[0]; // Just use the single batch directly
    }
    console.debug(`Stacked result shape: ${result.shape}`);

    // Rearrange dimensions to match expected output
    // From: [batch, channels, freqBins, frames, 2]
    // To: [batch, channels, 2, freqBins, frames]
    const permuteDims = result.shape.length === 5
      ? [0, 1, 4, 2, 3]  // With batch dimension
      : [0, 3, 1, 2];    // Without batch dimension

    const permuted = tf.transpose(result, permuteDims);
    console.debug(`Permuted shape: ${permuted.shape}`);

    // Merge channels and complex dimensions
    // From: [batch, channels, 2, freqBins, frames]
    // To: [batch, channels*2, freqBins, frames]
    let reshapeSize;
    if (result.shape.length === 5) {
      reshapeSize = [batchSize, channelCount * 2, permuted.shape[3], permuted.shape[4]];
    } else {
      reshapeSize = [channelCount * 2, permuted.shape[2], permuted.shape[3]];
    }
    const reshaped = permuted.reshape(reshapeSize);
    console.debug(`Reshaped: ${reshaped.shape}`);

    // Slice to requested frequency dimension
    const freqDimIndex = result.shape.length === 5 ? 2 : 1;
    const freqStart = 0;
    const freqSize = Math.min(this.dimF, reshaped.shape[freqDimIndex]);

    const freqSliceStart = Array(reshaped.shape.length).fill(0);
    const freqSliceSize = Array(reshaped.shape.length).fill(-1);
    freqSliceStart[freqDimIndex] = freqStart;
    freqSliceSize[freqDimIndex] = freqSize;

    const sliced = reshaped.slice(freqSliceStart, freqSliceSize);
    console.debug(`Final sliced shape: ${sliced.shape}`);

    // Clean up
    tensor.dispose();
    result.dispose();
    permuted.dispose();
    reshaped.dispose();
    outputs.forEach(o => o.dispose());

    return sliced;
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

  inverse(input_tensor: tf.Tensor4D): tf.Tensor {

    // Transfer the pre-defined Hann window tensor to the same device as the input tensor.
    const stft_window = this.hannWindow;

    const [batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins] = this.calculate_inverse_dimensions(input_tensor);

    const padded_tensor = this.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins);

    const complex_tensor = this.prepare_for_istft(padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim);

    // Perform the Inverse Short-Time Fourier Transform (ISTFT).
    // Note: TensorFlow.js doesn't have istft, so we need to use irfft and implement overlap-add
    const time_signal = tf.signal.irfft(complex_tensor);

    // Apply overlap-add reconstruction
    const num_frames = time_signal.shape[1];
    const frame_length = time_signal.shape[2];
    const num_samples = (num_frames - 1) * this.hopLength + frame_length;

    // Initialize output buffer
    const batch_channel_size = time_signal.shape[0];
    const outputBuffer = tf.buffer([batch_channel_size, num_samples]);
    const windowArray = stft_window.arraySync() as number[];
    const normBuffer = tf.buffer([batch_channel_size, num_samples]);

    const timeSignalArray = time_signal.arraySync() as number[][][];

    // Perform overlap-add
    for (let bc = 0; bc < batch_channel_size; bc++) {
      for (let t = 0; t < num_frames; t++) {
        const offset = t * this.hopLength;
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

    const result = final_output;

    // Clean up tensors
    input_tensor.dispose();
    padded_tensor.dispose();
    complex_tensor.dispose();
    time_signal.dispose();
    outputArray.dispose();
    normArray.dispose();
    normalized.dispose();

    return result;
  }
}
