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
    this.debugTensorStats(reshapedTensor, "stft: reshapedTensor");

    // Get STFT result in PyTorch format
    const stftOutput = this.torchStyleStft(
      reshapedTensor,
      this.nFft,
      this.hopLength,
      this.hannWindow,
      true  // center=true to match PyTorch
    );
    this.debugTensorStats(stftOutput, "stft: torchStyleStft output");

    // Rearrange the dimensions just like in the Python code
    // From: [batch*channels, freq_bins, time_frames, real/imag] 
    // To: [batch*channels, real/imag, freq_bins, time_frames]
    const permutedStftOutput = tf.transpose(stftOutput, [0, 3, 1, 2]);
    this.debugTensorStats(permutedStftOutput, "stft: permutedStftOutput");

    // Now reshape to restore original batch and channel dimensions
    // From: [batch*channels, real/imag, freq_bins, time_frames]
    // First: [...batch_dimensions, channel_dim, real/imag, freq_bins, time_frames]
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
  ): tf.Tensor { // [batch size, freq_bins, time_frames, real/imag]
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
      this.debugTensorStats(flatTensor, "torchStyleStft: flatTensor");

      // Process each flattened batch*channel
      const batchChannelSize = flatTensor.shape[0];
      const results = [];

      for (let bc = 0; bc < batchChannelSize; bc++) {
        // Extract this batch*channel as a 1D tensor
        const signal = flatTensor.slice([bc, 0], [1, -1]).reshape([timeLength]);
        this.debugTensorStats(signal, "torchStyleStft: flatTensor slice");

        // Apply padding if center=true (like PyTorch)
        let paddedSignal = signal;
        if (center) {
          const padSize = Math.floor(nFft / 2);
          paddedSignal = tf.mirrorPad(signal, [[padSize, padSize]], 'reflect');
        }
        this.debugTensorStats(paddedSignal, "torchStyleStft: paddedSignal");


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
    realPart: tf.Tensor, // [batch, freq, time]
    imagPart: tf.Tensor, // [batch, freq, time]
    nFft: number,
    hopLength: number,
    window: tf.Tensor,
    center: boolean = true,
    originalLength?: number
  ): tf.Tensor {
    console.debug(`ISTFT input shape: real=${realPart.shape}, imag=${imagPart.shape}, originalLength=${originalLength}`);

    return tf.tidy(() => {
      const channelCount = realPart.shape[0];
      const freqBins = realPart.shape[1];
      const timeFrames = realPart.shape[2];
      const results = [];

      for (let ch = 0; ch < channelCount; ch++) {
        // Extract this channel's real and imaginary parts
        const channelReal = realPart.slice([ch, 0, 0], [1, -1, -1]).squeeze([0]);
        const channelImag = imagPart.slice([ch, 0, 0], [1, -1, -1]).squeeze([0]);

        // Transpose from [freq, time] to [time, freq] for IRFFT
        const realT = tf.transpose(channelReal);
        const imagT = tf.transpose(channelImag);

        // Apply conjugate for IFFT (negate imaginary part)
        const conjImagT = tf.neg(imagT);

        // Create complex tensor for IRFFT
        const complexTensor = tf.complex(realT, conjImagT);

        // Apply IRFFT to get time-domain frames
        const frames = tf.spectral.irfft(complexTensor);
        console.debug(`IRFFT output frames shape: ${frames.shape}`);

        // Apply window to each frame
        const frameLength = frames.shape[1] || nFft;
        const windowTensor = window.slice(0, frameLength);
        const reshapedWindow = windowTensor.reshape([1, frameLength]);
        const windowedFrames = frames.mul(reshapedWindow);

        // Calculate output length
        const outputLength = (timeFrames - 1) * hopLength + nFft;
        console.log(`ISTFT: timeFrames=${timeFrames}, hopLength=${hopLength}, nFft=${nFft}, calculated outputLength=${outputLength}`);

        // Create output tensor and normalization tensor
        const outputTensor = tf.buffer([outputLength]);
        const normTensor = tf.buffer([outputLength]);

        // Extract frames data
        const framesData = windowedFrames.arraySync();
        const windowData = windowTensor.arraySync();

        // Perform overlap-add using vectorized operations
        for (let t = 0; t < timeFrames; t++) {
          const offset = t * hopLength;
          const frame = framesData[t];

          // We can optimize this by using tensor slices, but for now
          // we'll use the buffer approach for clarity
          for (let i = 0; i < Math.min(frame.length, frameLength); i++) {
            if (offset + i < outputLength) {
              outputTensor.set(outputTensor.get(offset + i) + frame[i], offset + i);
              normTensor.set(normTensor.get(offset + i) + windowData[i] * windowData[i], offset + i);
            }
          }
        }
        console.log(`ISTFT: actual output length=${outputTensor.toTensor().shape[0]}`);

        // Convert buffers to tensors
        const output = outputTensor.toTensor();
        const norm = normTensor.toTensor();

        // Apply normalization
        const normalized = tf.div(output, tf.add(norm, tf.scalar(1e-8)));

        // Remove padding if center was true
        let result = normalized;
        if (center) {
          const padSize = Math.floor(nFft / 2);
          const contentAfterCenterRemovalLength = outputLength - (2 * padSize);
          if (normalized.shape[0] >= padSize + contentAfterCenterRemovalLength && contentAfterCenterRemovalLength >=0) {
            result = normalized.slice([padSize], [contentAfterCenterRemovalLength]);
            console.log(`ISTFT: Removed center padding. OLA output: ${outputLength}, Content length: ${contentAfterCenterRemovalLength}. Result shape: ${result.shape[0]}`);
          } else {
            const actualSliceLength = Math.max(0, contentAfterCenterRemovalLength);
            if (normalized.shape[0] >= padSize + actualSliceLength) {
                 result = normalized.slice([padSize], [actualSliceLength]);
                 console.warn(`ISTFT: Problematic slice for center padding. OLA: ${outputLength}, pad: ${padSize}, calc content: ${contentAfterCenterRemovalLength}. Sliced ${actualSliceLength}.`);
            } else {
                 result = tf.tensor1d([]); // Ensure it's 1D for subsequent operations
                 console.error(`ISTFT: Normalized tensor too short for center slice. Shape: ${normalized.shape[0]}, pad: ${padSize}, calc content: ${contentAfterCenterRemovalLength}`);
            }
          }
        }
        // Else result is 'normalized'

        // Adjust to originalLength if provided
        if (originalLength !== undefined) {
          const currentLength = result.shape[0];
          if (currentLength > originalLength) {
            result = result.slice([0], [originalLength]);
            console.log(`ISTFT: Truncated from ${currentLength} to originalLength ${originalLength}.`);
          } else if (currentLength < originalLength) {
            result = tf.pad(result, [[0, originalLength - currentLength]]);
            console.log(`ISTFT: Padded from ${currentLength} to originalLength ${originalLength}.`);
          }
        }
        results.push(result);
      }

      // Stack channels
      const stackedResults = tf.stack(results);
      console.log(`ISTFT: Final stacked results shape: ${stackedResults.shape}, originalLength provided: ${originalLength}`);
      return stackedResults;
    });
  }

  /**
   * Prepares tensors for ISTFT without creating a complex tensor
   * Returns real and imaginary parts separately
   */
  private prepareForIstft(
    paddedTensor: tf.Tensor,
    batchDimensions: number[],
    channelDim: number,
    numFreqBins: number,
    timeDim: number
  ): { real: tf.Tensor, imag: tf.Tensor } {
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

    console.debug(`Real part shape: ${realPart.shape}, Imag part shape: ${imagPart.shape}`);

    // Clean up intermediate tensors
    reshapedTensor.dispose();
    flattenedTensor.dispose();
    permutedTensor.dispose();

    return { real: realPart, imag: imagPart };
  }

  inverse(inputTensor: tf.Tensor, originalLength?: number): tf.Tensor {
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

    // Prepare for ISTFT - get real and imaginary parts
    const { real, imag } = this.prepareForIstft(
      paddedTensor, batchDimensions, channelDim, numFreqBins, timeDim
    );

    console.debug(`Prepared real shape: ${real.shape}, imag shape: ${imag.shape}`);

    // Perform ISTFT with separate real and imaginary parts
    const result = this.istft(
      real,
      imag,
      this.nFft,
      this.hopLength,
      this.hannWindow,
      true, // center=true to match PyTorch
      originalLength // Pass originalLength here
    );
    const istftResult = result.expandDims(0); // result from istft should now respect originalLength
    console.debug(`ISTFT direct result expanded shape: ${istftResult.shape}`);

    // The istft method should now handle originalLength.
    // The truncation/padding logic here becomes redundant if originalLength was passed to istft.
    // However, if originalLength was NOT passed to istft (e.g. STFT.inverse called with 1 arg),
    // this block could still be relevant, but originalLength would be undefined.
    // For this specific subtask, we assume originalLength is always passed from inverse to istft.

    // Clean up
    paddedTensor.dispose();
    real.dispose();
    imag.dispose();
    result.dispose(); // this is the tensor from istft *before* expandDims

    return istftResult;
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


  /**
  * Debug utility to check tensor stats including NaN/Inf detection
  */
  private debugTensorStats(tensor: tf.Tensor, name: string): void {
    const stats = tf.tidy(() => {
      const abs = tf.abs(tensor);
      const max = tf.max(abs);
      const mean = tf.mean(abs);
      const nonZero = tf.sum(tf.cast(tf.greater(abs, 1e-8), 'int32'));
      const total = tf.scalar(tensor.size);

      // NaN and Inf detection
      const isNaN = tf.isNaN(tensor);
      const isInf = tf.isInf(tensor);
      const nanCount = tf.sum(tf.cast(isNaN, 'int32'));
      const infCount = tf.sum(tf.cast(isInf, 'int32'));

      return {
        max: max.dataSync()[0],
        mean: mean.dataSync()[0],
        nonZeroCount: nonZero.dataSync()[0],
        totalCount: total.dataSync()[0],
        nanCount: nanCount.dataSync()[0],
        infCount: infCount.dataSync()[0],
        shape: tensor.shape
      };
    });

    const nonZeroPercent = (stats.nonZeroCount / stats.totalCount * 100).toFixed(2);
    const status = stats.nanCount > 0 ? "🚨 NaN" : stats.infCount > 0 ? "⚠️ Inf" : stats.max < 1e-8 ? "🤔 Zero" : "✅ OK";

    console.log(`${status} ${name}: shape=${stats.shape}, max=${stats.max.toFixed(6)}, mean=${stats.mean.toFixed(6)}, nonZero=${stats.nonZeroCount}/${stats.totalCount} (${nonZeroPercent}%)`);

    if (stats.nanCount > 0) {
      console.error(`   └─ Contains ${stats.nanCount} NaN values!`);
    }
    if (stats.infCount > 0) {
      console.error(`   └─ Contains ${stats.infCount} Inf values!`);
    }
    if (stats.max < 1e-8 && stats.nanCount === 0) {
      console.warn(`   └─ Tensor appears to be essentially zero`);
    }
  }
}
