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
    console.debug(`STFT forward input tensor shape: ${tensor.shape}`);

    // Extract batch dimensions (all dimensions except the last two which are channel and time).
    const batchShape = tensor.shape.slice(0, -2);
    const batchDimensions = batchShape.length > 0 ? batchShape : [1];

    // Extract channel and time dimensions (last two dimensions of the tensor).
    const channelDim = tensor.shape[tensor.shape.length - 2];
    const timeDim = tensor.shape[tensor.shape.length - 1];

    console.debug(`Input dimensions: batch=${batchDimensions}, channels=${channelDim}, time=${timeDim}`);

    // Reshape the tensor to merge batch and channel dimensions for STFT processing.
    const reshapedTensor = tensor.reshape([-1, timeDim]);
    console.debug(`Reshaped tensor: ${reshapedTensor.shape}`);

    // Get STFT result in PyTorch format
    const stftOutput = this.torchStyleStft(
      reshapedTensor,
      this.nFft,
      this.hopLength,
      this.hannWindow,
      true  // center=true to match PyTorch
    );

    console.debug(`Raw STFT output shape: ${stftOutput.shape}`);

    // Rearrange the dimensions just like in the Python code
    // From: [batch*channels, freq_bins, time_frames, 2] 
    // To: [batch*channels, 2, freq_bins, time_frames]
    const permutedStftOutput = tf.transpose(stftOutput, [0, 3, 1, 2]);
    console.debug(`Permuted STFT output shape: ${permutedStftOutput.shape}`);

    // Now reshape to restore original batch and channel dimensions
    // From: [batch*channels, 2, freq_bins, time_frames]
    // First: [...batch_dimensions, channel_dim, 2, freq_bins, time_frames]
    // Then: [...batch_dimensions, channel_dim*2, freq_bins, time_frames]

    // Calculate the new shape dimensions
    const freqBins = permutedStftOutput.shape[2];
    const timeFrames = permutedStftOutput.shape[3];

    // First reshape to separate batch and channel dimensions
    const intermediateShape = [...batchDimensions, channelDim, 2, freqBins, timeFrames];
    const intermediate = permutedStftOutput.reshape(intermediateShape);
    console.debug(`Intermediate reshape: ${intermediate.shape}`);

    // Second reshape to merge channel and complex dimensions
    const finalShape = [...batchDimensions, channelDim * 2, freqBins, timeFrames];
    const finalOutput = intermediate.reshape(finalShape);
    console.debug(`Final output shape: ${finalOutput.shape}`);

    // Slice to retain only the required frequency dimension (dim_f)
    const freqDimIndex = finalOutput.shape.length - 2;
    const sliceStart = Array(finalOutput.shape.length).fill(0);
    const sliceSize = Array(finalOutput.shape.length).fill(-1);
    sliceSize[freqDimIndex] = Math.min(this.dimF, finalOutput.shape[freqDimIndex]);

    const slicedOutput = finalOutput.slice(sliceStart, sliceSize);
    console.debug(`Sliced output shape: ${slicedOutput.shape}`);

    // Clean up tensors
    reshapedTensor.dispose();
    stftOutput.dispose();
    permutedStftOutput.dispose();
    intermediate.dispose();
    finalOutput.dispose();

    return slicedOutput;
  }

  private torchStyleStft(
    tensor: tf.Tensor,
    nFft: number,
    hopLength: number,
    window: tf.Tensor,
    center: boolean = true
  ): tf.Tensor {
    console.debug(`Torch-style STFT input shape: ${tensor.shape}`);

    return tf.tidy(() => {
      // Get tensor dimensions
      const shape = tensor.shape;
      const rank = shape.length;

      // Extract dimensions
      const timeLength = shape[rank - 1];
      const channelDim = shape[rank - 2];
      const batchShape = shape.slice(0, -2);

      // Reshape to flatten ALL dimensions except time
      // This is equivalent to reshaped_tensor = input_tensor.reshape([-1, time_dim])
      const flatTensor = tensor.reshape([-1, timeLength]);
      console.debug(`Flattened tensor shape: ${flatTensor.shape}`);

      // Process each flattened batch*channel
      const batchChannelSize = flatTensor.shape[0];
      const results = [];

      for (let bc = 0; bc < batchChannelSize; bc++) {
        // Extract this batch*channel as a 1D tensor
        const signal = flatTensor.slice([bc, 0], [1, -1]).reshape([timeLength]);

        // Apply padding if center=true (like PyTorch)
        let paddedSignal = signal;
        if (center) {
          const padSize = Math.floor(nFft / 2);
          paddedSignal = tf.pad(signal, [[padSize, padSize]], 'reflect');
        }

        // Apply STFT
        const stftResult = tf.signal.stft(
          paddedSignal,
          nFft,
          hopLength,
          nFft,
          () => window
        );

        // Extract real and imaginary components
        const real = tf.real(stftResult);
        const imag = tf.imag(stftResult);

        // TF returns [time_frames, freq_bins], but PyTorch expects [freq_bins, time_frames]
        // Transpose to match PyTorch's dimension order
        const realTransposed = tf.transpose(real);
        const imagTransposed = tf.transpose(imag);

        // Stack real and imag as the last dimension (matches PyTorch's format)
        // This gives us [freq_bins, time_frames, 2]
        const complexStacked = tf.stack([realTransposed, imagTransposed], -1);
        results.push(complexStacked);

        // Clean up
        signal.dispose();
        paddedSignal.dispose();
        stftResult.dispose();
        real.dispose();
        imag.dispose();
        realTransposed.dispose();
        imagTransposed.dispose();
      }

      // Stack all results
      // This should give [batch*channels, freq_bins, time_frames, 2]
      const stacked = tf.stack(results, 0);
      console.debug(`Final stacked STFT result: ${stacked.shape}`);

      return stacked;
    });
  }


  // Helper methods for inverse STFT
  private padFrequencyDimension(
    inputTensor: tf.Tensor,
    batchDimensions: number[],
    channelDim: number,
    freqDim: number,
    timeDim: number,
    numFreqBins: number
  ): tf.Tensor {
    console.debug(`Padding frequency dimension from ${freqDim} to ${numFreqBins}`);

    // Create padding tensor for frequency dimension
    const freqPadding = tf.zeros([...batchDimensions, channelDim, numFreqBins - freqDim, timeDim]);

    // Concatenate along frequency dimension (-2)
    const padded = tf.concat([inputTensor, freqPadding], inputTensor.shape.length - 2);

    // Clean up
    freqPadding.dispose();

    return padded;
  }

  //
  // INVERSE
  //

  /**
 * PyTorch-compatible ISTFT implementation
 * Matches the behavior of torch.istft
 * Input: (batches, freqBins, timeFrames)
 * Output shape: (batches, channels, length)
 */
  /**
 * Inverse Short-Time Fourier Transform implementation
 * Matches the behavior of torch.istft
 * 
 * @param complexTensor Complex tensor of shape [batch, freq_bins, time_frames]
 * @param nFft FFT size
 * @param hopLength Hop length between frames
 * @param window Window function tensor of length nFft
 * @param center Whether the input was padded for centered frames
 * @returns Time domain signal of shape [batch, signal_length]
 */
  private istft(
    complexTensor: tf.Tensor, // [batch, freq, time]
    nFft: number,
    hopLength: number,
    window: tf.Tensor,
    center: boolean = true
  ): tf.Tensor {
    console.debug(`ISTFT input shape: ${complexTensor.shape}`);

    return tf.tidy(() => {
      const batchSize = complexTensor.shape[0];
      const freqBins = complexTensor.shape[1];
      const timeFrames = complexTensor.shape[2];
      const results = [];

      // Process each channel separately
      for (let b = 0; b < batchSize; b++) {
        // Get this channel
        const channelTensor = complexTensor.slice([b, 0, 0], [1, -1, -1]).squeeze([0]);
        console.debug(`Processing channel ${b}, shape: ${channelTensor.shape}`);

        // Split real and imaginary parts
        const realPart = tf.real(channelTensor);
        const imagPart = tf.imag(channelTensor);

        // Create a proper complex tensor for one channel
        const channelComplex = tf.complex(realPart, imagPart);

        // Apply IRFFT to this single channel
        console.debug(`IRFFT input shape for channel ${b}: ${channelComplex.shape}`);

        const frames = tf.spectral.irfft(channelComplex);
        console.debug(`IRFFT output shape for channel ${b}: ${frames.shape}`);

        // Apply window function and perform overlap-add
        const frameCount = frames.shape[0];
        const frameLength = frames.shape[1];

        // Reshape window for broadcasting
        const windowTensor = window.slice([0], [frameLength]);
        const windowedFrames = tf.mul(frames, windowTensor);

        // Manual overlap-add reconstruction
        const outputLength = (timeFrames - 1) * hopLength + nFft;
        const outputBuffer = new Float32Array(outputLength).fill(0);
        const normBuffer = new Float32Array(outputLength).fill(0);

        // Get data as JavaScript arrays
        const framesData = windowedFrames.arraySync() as number[][];
        const windowData = windowTensor.arraySync() as number[];

        // Perform overlap-add
        for (let t = 0; t < frameCount; t++) {
          const offset = t * hopLength;
          for (let i = 0; i < frameLength; i++) {
            if (offset + i < outputLength) {
              outputBuffer[offset + i] += framesData[t][i];
              normBuffer[offset + i] += windowData[i] * windowData[i];
            }
          }
        }

        // Normalize
        for (let i = 0; i < outputLength; i++) {
          if (normBuffer[i] > 1e-8) {
            outputBuffer[i] /= normBuffer[i];
          }
        }

        // Remove center padding
        let resultBuffer = outputBuffer;
        if (center) {
          const padSize = Math.floor(nFft / 2);
          if (outputLength > 2 * padSize) {
            resultBuffer = outputBuffer.slice(padSize, outputLength - padSize);
          }
        }

        // Convert to tensor
        results.push(tf.tensor1d(resultBuffer));

        // Clean up
        realPart.dispose();
        imagPart.dispose();
        channelComplex.dispose();
        frames.dispose();
        windowedFrames.dispose();
        windowTensor.dispose();
      }

      // Stack all channels
      return tf.stack(results);
    });
  }

  private calculateInverseDimensions(inputTensor: tf.Tensor): [number[], number, number, number, number] {
    // Extract batch dimensions and frequency-time dimensions
    const shape = inputTensor.shape;
    const rank = shape.length;

    // Last 3 dimensions are channel, freq, time
    const batchDimensions = shape.slice(0, rank - 3);
    const channelDim = shape[rank - 3];
    const freqDim = shape[rank - 2];
    const timeDim = shape[rank - 1];

    // Calculate number of frequency bins for inverse STFT
    const numFreqBins = Math.floor(this.nFft / 2) + 1;

    return [batchDimensions, channelDim, freqDim, timeDim, numFreqBins];
  }


  private prepareForIstft(
    paddedTensor: tf.Tensor,
    batchDimensions: number[],
    channelDim: number,
    numFreqBins: number,
    timeDim: number
  ): tf.Tensor {
    console.debug(`Preparing for ISTFT with shape: ${paddedTensor.shape}`);

    // Reshape to separate real and imaginary parts
    // From: [...batchDimensions, channelDim, numFreqBins, timeDim]
    // To: [...batchDimensions, channelDim/2, 2, numFreqBins, timeDim]
    const intermediateDims = [...batchDimensions, channelDim / 2, 2, numFreqBins, timeDim];
    const reshapedTensor = paddedTensor.reshape(intermediateDims);

    // Flatten batch dimensions
    // From: [...batchDimensions, channelDim/2, 2, numFreqBins, timeDim]
    // To: [batchSize*channelDim/2, 2, numFreqBins, timeDim]
    const batchSize = batchDimensions.length === 0 ? 1 :
      batchDimensions.reduce((a, b) => a * b, 1);
    const flattenedTensor = reshapedTensor.reshape([batchSize * channelDim / 2, 2, numFreqBins, timeDim]);

    // Permute dimensions to match PyTorch format for ISTFT
    // From: [batchSize*channelDim/2, 2, numFreqBins, timeDim]
    // To: [batchSize*channelDim/2, numFreqBins, timeDim, 2]
    const permutedTensor = flattenedTensor.transpose([0, 2, 3, 1]);

    // Extract real and imaginary parts
    const realPart = permutedTensor.slice([0, 0, 0, 0], [-1, -1, -1, 1]).squeeze([-1]);
    const imagPart = permutedTensor.slice([0, 0, 0, 1], [-1, -1, -1, 1]).squeeze([-1]);

    // Create a proper TF complex tensor
    const complexTensor = tf.complex(realPart, imagPart);

    // Clean up intermediate tensors
    reshapedTensor.dispose();
    flattenedTensor.dispose();
    permutedTensor.dispose();
    realPart.dispose();
    imagPart.dispose();

    return complexTensor;
  }

  inverse(inputTensor: tf.Tensor): tf.Tensor {
    console.debug(`STFT inverse input tensor shape: ${inputTensor.shape}`);

    // Calculate dimensions
    const [batchDimensions, channelDim, freqDim, timeDim, numFreqBins] =
      this.calculateInverseDimensions(inputTensor);

    console.debug(`Dimensions: batch=${batchDimensions}, channels=${channelDim}, freq=${freqDim}, time=${timeDim}, numFreqBins=${numFreqBins}`);

    // Pad frequency dimension if needed
    const paddedTensor = this.padFrequencyDimension(
      inputTensor, batchDimensions, channelDim, freqDim, timeDim, numFreqBins
    );

    console.debug(`Padded tensor shape: ${paddedTensor.shape}`);

    // Prepare for ISTFT
    const complexTensor = this.prepareForIstft(
      paddedTensor, batchDimensions, channelDim, numFreqBins, timeDim
    );

    console.debug(`Complex tensor shape: ${complexTensor.shape}`);

    // Perform ISTFT
    const istftResult = this.istft(
      complexTensor,
      this.nFft,
      this.hopLength,
      this.hannWindow,
      true  // center=true to match PyTorch
    );

    console.debug(`ISTFT result shape: ${istftResult.shape}`);

    // Reshape to restore original batch and channel dimensions
    // From: [batchSize*channelDim/2, timeLength]
    // To: [...batchDimensions, 2, timeLength]
    const batchSize = batchDimensions.length === 0 ? 1 :
      batchDimensions.reduce((a, b) => a * b, 1);

    // Check if reshaping is needed
    let finalOutput = istftResult;

    if (istftResult.shape[0] === batchSize * channelDim / 2) {
      // If we have multiple batches/channels, reshape
      const reshapeSize = [...batchDimensions, 2, istftResult.shape[1]];
      finalOutput = istftResult.reshape(reshapeSize);
    }

    console.debug(`Final output shape: ${finalOutput.shape}`);

    // Clean up
    paddedTensor.dispose();
    complexTensor.dispose();
    if (finalOutput !== istftResult) {
      istftResult.dispose();
    }

    return finalOutput;
  }
}
