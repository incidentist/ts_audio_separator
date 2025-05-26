import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

import { CommonSeparator, SeparatorConfig } from '../common_separator';
import { AudioUtils } from '../../utils/audio_utils';
import { STFT } from './stft';

export interface MDXArchConfig {
  segmentSize?: number;
  overlap?: number;
  batchSize?: number;
  hopLength?: number;
  enableDenoise?: boolean;
}

export class MDXSeparator extends CommonSeparator {
  // Architecture-specific configuration
  private segmentSize: number;
  private overlap: number;
  private batchSize: number;
  private hopLength: number;
  private enableDenoise: boolean;

  // Model-specific parameters from model data
  private compensate: number;
  private dimF: number;
  private dimT: number;
  private nFft: number;
  private configYaml?: string;

  // ONNX model and runtime
  private model?: ort.InferenceSession;

  // Processing parameters
  private nBins: number = 0;
  private trim: number = 0;
  private chunkSize: number = 0;
  private genSize: number = 0;
  private stft?: STFT;


  // State during processing
  private primarySource?: tf.Tensor;
  private secondarySource?: tf.Tensor;
  private audioFilePath?: string;
  private audioFileBase?: string;

  constructor(commonConfig: SeparatorConfig, archConfig: MDXArchConfig) {
    super(commonConfig);

    // Initialize architecture-specific parameters
    this.segmentSize = archConfig.segmentSize || 256;
    this.overlap = archConfig.overlap || 0.25;
    this.batchSize = archConfig.batchSize || 1;
    this.hopLength = archConfig.hopLength || 1024;
    this.enableDenoise = archConfig.enableDenoise || false;

    this.debug(`MDX arch params: batchSize=${this.batchSize}, segmentSize=${this.segmentSize}`);
    this.debug(`MDX arch params: overlap=${this.overlap}, hopLength=${this.hopLength}, enableDenoise=${this.enableDenoise}`);

    // Initialize model-specific parameters
    this.compensate = this.modelData.compensate || 1.0;
    this.dimF = this.modelData.mdx_dim_f_set || 3072;
    this.dimT = Math.pow(2, this.modelData.mdx_dim_t_set || 8);
    this.nFft = this.modelData.mdx_n_fft_scale_set || 7680;
    this.configYaml = this.modelData.config_yaml;

    this.debug(`MDX arch params: compensate=${this.compensate}, dimF=${this.dimF}, dimT=${this.dimT}, nFft=${this.nFft}`);
    this.debug(`MDX arch params: configYaml=${this.configYaml}`);
  }

  async loadModel(): Promise<void> {
    this.debug('Loading ONNX model for inference...');

    // Download model if not already downloaded
    if (!this.model) {
      await this.downloadModel();
    }

    // Initialize ONNX runtime
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      logSeverityLevel: this.logLevel === 'debug' ? 0 : 3,
    };

    // Configure WASM paths
    const wasmVersion = '1.22.0';
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${wasmVersion}/dist/`;

    // Load the model
    this.model = await ort.InferenceSession.create(this.modelPath, sessionOptions);
    this.debug('Model loaded successfully using ONNXruntime inferencing session.');

    // Initialize processing parameters
    this.initializeModelSettings();
  }

  private async downloadModel(): Promise<void> {
    this.info('Downloading UVR_MDXNET_KARA_2 model from Hugging Face...');

    const modelUrl = 'https://huggingface.co/AI4future/RVC/resolve/main/UVR_MDXNET_KARA_2.onnx';

    try {
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to download model: ${response.statusText}`);
      }

      const buffer = await response.arrayBuffer();

      // Store the model data
      // In a real implementation, you might want to cache this to avoid re-downloading
      this.modelPath = URL.createObjectURL(new Blob([buffer]));

      this.info('Model downloaded successfully');
    } catch (error) {
      this.error(`Error downloading model: ${error}`);
      throw error;
    }
  }

  private initializeModelSettings(): void {
    this.debug('Initializing model settings...');

    // n_bins is half the FFT size plus one
    this.nBins = Math.floor(this.nFft / 2) + 1;

    // trim is half the FFT size
    this.trim = Math.floor(this.nFft / 2);

    // chunk_size calculation from Python: self.hop_length * (self.segment_size - 1)
    // But we need to ensure it produces exactly segment_size frames when STFT is applied
    // STFT frames formula: frames = (samples - nFft) / hopLength + 1
    // Rearranging: samples = (frames - 1) * hopLength + nFft
    // For segmentSize frames: samples = (segmentSize - 1) * hopLength + nFft
    // this.chunkSize = (this.segmentSize - 1) * this.hopLength + this.nFft;
    this.chunkSize = 261120; // 256 * 1024 + 7680


    // gen_size is the chunk size minus twice the trim size
    this.genSize = this.chunkSize - (2 * this.trim);

    // Initialize STFT
    this.stft = new STFT(this.nFft, this.hopLength, this.dimF);

    this.debug(`Model input params: nFft=${this.nFft} hopLength=${this.hopLength} dimF=${this.dimF}`);
    this.debug(`Model settings: nBins=${this.nBins}, trim=${this.trim}, chunkSize=${this.chunkSize}, genSize=${this.genSize}`);

    // Verify chunk size will produce correct frames
    const expectedFrames = Math.floor((this.chunkSize - this.nFft) / this.hopLength) + 1;
    this.debug(`Expected frames from chunk size: ${expectedFrames}, Target: ${this.segmentSize}`);
  }

  async separate(audioFilePath: string, customOutputNames?: Record<string, string>): Promise<string[]> {
    this.audioFilePath = audioFilePath;
    this.audioFileBase = audioFilePath.split('/').pop()?.split('.')[0] || 'audio';

    // Load audio file
    this.debug(`Loading audio file ${audioFilePath}...`);
    const audioChannels: Float32Array<ArrayBufferLike>[] = await AudioUtils.loadAudioFile(audioFilePath, this.sampleRate);

    this.debug(`Converting audio to tensor...`);
    let mixTensor = tf.tensor(audioChannels);


    // Prepare the mix for processing
    this.debug(`Preparing mix for input audio file ${this.audioFilePath}...`);
    const mix = await this.prepareMix(mixTensor);

    this.debug('Normalizing mix before demixing...');
    const normalizedMix = await this.normalize(mix, this.normalizationThreshold, this.amplificationThreshold);

    // Start the demixing process
    const source = await this.demix(normalizedMix);
    this.debug('Demixing completed.');

    // Initialize the list for output files
    const outputFiles: string[] = [];
    this.debug('Processing output files...');

    // Normalize and process the primary source if it's not already an array
    if (!this.primarySource) {
      this.debug('Normalizing primary source...');
      this.primarySource = await this.normalize(source, this.normalizationThreshold, this.amplificationThreshold);
    }

    // Process the secondary source if not already an array
    if (!this.secondarySource) {
      this.debug('Producing secondary source');

      if (this.invertUsingSpec) {
        throw Error('invertUsingSpec is not implemented');
        // For spectral inversion, we'd need to implement invertStem method
        // const rawMix = await this.demix(normalizedMix, true);
        // this.secondarySource = this.invertStem(rawMix, source);
      } else {
        this.debug('Inverting secondary stem by subtracting demixed stem from original mix');
        this.secondarySource = this.subtractChannels(normalizedMix, source);
      }
    }

    // Save and process the secondary stem if needed
    if (!this.outputSingleStem || this.outputSingleStem.toLowerCase() === this.secondaryStemName.toLowerCase()) {
      const secondaryPath = this.getStemOutputPath(this.secondaryStemName, customOutputNames);

      this.info(`Saving secondary ${this.secondaryStemName} stem to ${secondaryPath}...`);
      const secondaryUrl = await this.saveAudioOutput(secondaryPath, this.secondarySource!, this.secondaryStemName);
      outputFiles.push(secondaryUrl);
    }

    // Save and process the primary stem if needed
    if (!this.outputSingleStem || this.outputSingleStem.toLowerCase() === this.primaryStemName.toLowerCase()) {
      const primaryPath = this.getStemOutputPath(this.primaryStemName, customOutputNames);

      if (!this.primarySource) {
        this.primarySource = source;
      }

      this.info(`Saving primary ${this.primaryStemName} stem to ${primaryPath}...`);
      const primaryUrl = await this.saveAudioOutput(primaryPath, this.primarySource, this.primaryStemName);
      outputFiles.push(primaryUrl);
    }

    return outputFiles;
  }

  // Method to prepare mix for processing
  async prepareMix(audioData: tf.Tensor): Promise<tf.Tensor2D> {
    this.debug(`Preparing mix for input audio...`);
    return super.prepareMix(audioData);
  }

  // Method to normalize audio
  async normalize(
    wave: tf.Tensor,            // tf.Tensor1D | tf.Tensor2D
    maxPeak = 1.0,
    minPeak?: number            // optional
  ): Promise<tf.Tensor> {
    this.debug('Normalizing audio...');
    return super.normalize(wave, maxPeak, minPeak);
  }

  private async demix(mix: tf.Tensor2D, isMatchMix: boolean = false): Promise<tf.Tensor> {
    const start = performance.now();
    this.debugTensorStats(mix, `demix input (isMatchMix=${isMatchMix})`);
    this.initializeModelSettings();

    // Preserves the original mix for later use
    const orgMix = mix;
    this.debug(`Original mix stored. Shape: ${mix.shape}`);

    // Handling different chunk sizes and overlaps based on the matching requirement
    let chunkSize: number;
    let overlap: number;

    if (isMatchMix) {
      // Sets a smaller chunk size specifically for matching the mix
      chunkSize = this.hopLength * (this.segmentSize - 1);
      // Sets a small overlap for the chunks
      overlap = 0.02;
      this.debug(`Chunk size for matching mix: ${chunkSize}, Overlap: ${overlap}`);
    } else {
      // Uses the regular chunk size defined in model settings
      chunkSize = this.chunkSize;
      // Uses the overlap specified in the model settings
      overlap = this.overlap;
      this.debug(`Standard chunk size: ${chunkSize}, Overlap: ${overlap}`);
    }

    // Calculate the generated size after subtracting the trim from both ends of the chunk
    const genSize = chunkSize - 2 * this.trim;
    this.debug(`Generated size calculated: ${genSize}`);

    // Calculate padding to make the mix length a multiple of the generated size
    const mixLength = mix.shape[1];  // Assuming mix is [channels, time]
    const pad = genSize + this.trim - (mixLength % genSize);

    // Create zeros tensors for padding at beginning and end
    const leftPad = tf.zeros([2, this.trim]);  // 2 channels, trim samples
    const rightPad = tf.zeros([2, pad]);       // 2 channels, pad samples

    // Concatenate along time dimension (axis 1) to create padded mixture
    const mixture = tf.concat([leftPad, mix, rightPad], 1);
    this.debugTensorStats(mixture, `runModel: padded mixture`);

    // Calculate the step size for processing chunks based on the overlap
    const step = Math.floor((1 - overlap) * chunkSize);
    this.debug(`Step size for processing chunks: ${step} as overlap is set to ${overlap}.`);

    // Initialize arrays to store the results and to account for overlap
    const resultShape = [1, 2, mixture.shape[1]];
    let result = tf.zeros(resultShape);     // For accumulating processed chunks
    let divider = tf.zeros(resultShape);    // For tracking overlap counts

    this.debug(`Initialized result and divider tensors with shape: ${result.shape}`);

    // Initialize counters for processing chunks
    let total = 0;
    const mixtureLength = mixture.shape[1];

    // Calculate total chunks more accurately
    // We need at least one full chunk worth of data to process
    const effectiveLength = mixtureLength - chunkSize;
    const totalChunks = effectiveLength > 0 ? Math.floor(effectiveLength / step) + 1 : 1;

    this.debug(`Mixture length: ${mixtureLength}, Effective length for stepping: ${effectiveLength}`);
    this.debug(`Total chunks to process: ${totalChunks}`);

    // Process each chunk of the mixture
    for (let i = 0; i <= effectiveLength && i < mixtureLength; i += step) {
      total++;
      const start = i;
      const end = Math.min(i + chunkSize, mixtureLength);
      const actualChunkSize = end - start;

      // Skip if we don't have enough samples for a meaningful chunk
      if (actualChunkSize < this.nFft) {
        this.debug(`Skipping chunk ${total} - too small (${actualChunkSize} < ${this.nFft})`);
        break;
      }

      this.debug(`Processing chunk ${total}/${totalChunks}: Start ${start}, End ${end}, Size ${actualChunkSize}`);

      // Handle windowing for overlapping chunks
      let window: tf.Tensor | null = null;

      if (overlap !== 0) {
        // Create Hann window for the actual chunk size
        const windowBase = tf.signal.hannWindow(actualChunkSize);
        // Reshape to match the shape [1, 2, actualChunkSize]
        window = windowBase.expandDims(0).expandDims(0).tile([1, 2, 1]);
        windowBase.dispose();
        this.debug("Window applied to the chunk.");
      }

      // Extract the chunk from mixture
      let mixPart_ = mixture.slice([0, start], [2, actualChunkSize]);

      // Zero-pad if chunk is incomplete
      if (actualChunkSize < chunkSize) {
        const padSize = chunkSize - actualChunkSize;
        const padding = tf.zeros([2, padSize]);
        mixPart_ = tf.concat([mixPart_, padding], 1);
        padding.dispose();
        this.debug(`Padded incomplete chunk with ${padSize} zeros`);
      }

      // Add batch dimension
      const mixPart = mixPart_.expandDims(0);  // Shape: [1, 2, chunkSize]
      mixPart_.dispose();

      // Split into batches if needed
      const mixWaves = tf.split(mixPart, Math.ceil(mixPart.shape[0] / this.batchSize));
      const totalBatches = mixWaves.length;
      this.debug(`Mix part split into batches. Number of batches: ${totalBatches}`);

      let batchesProcessed = 0;
      for (const mixWave of mixWaves) {
        batchesProcessed += 1;
        this.debug(`Processing mix_wave batch ${batchesProcessed}/${totalBatches}`);

        // Run the model to separate the sources
        const tarWaves = await this.runModel(mixWave, isMatchMix);

        // Apply window and update result/divider
        const { result: newResult, divider: newDivider } =
          this.applyWindow(tarWaves, window, result, divider, start, end, actualChunkSize);

        // Update with new tensors and clean up old ones
        result.dispose();
        divider.dispose();
        result = newResult;
        divider = newDivider;

        // Clean up
        tarWaves.dispose();
        mixWave.dispose();
      }

      // Clean up window if it exists
      if (window) window.dispose();
      mixPart.dispose();
    }

    // Clean up padding tensors
    leftPad.dispose();
    rightPad.dispose();
    mixture.dispose();

    // Debug divider before division
    this.debugTensorStats(result, "Result tensor before division");
    this.debugTensorStats(divider, "Divider tensor before division");

    // Normalize the results by the divider to account for overlap
    // Add small epsilon to prevent division by zero
    const epsilon = 1e-8;
    const dividerPlusEpsilon = divider.add(epsilon);
    const tarWaves = tf.div(result, dividerPlusEpsilon);

    this.debug(`Normalized tar_waves shape: ${tarWaves.shape}`);

    // Clean up intermediate tensors
    result.dispose();
    divider.dispose();
    dividerPlusEpsilon.dispose();

    // Extract the valid portion (remove padding)
    // The valid portion starts at trim and ends at trim + original mix length
    const validStart = this.trim;
    const validEnd = this.trim + mixLength;

    this.debug(`Extracting valid portion: start=${validStart}, end=${validEnd}, length=${mixLength}`);

    // Slice to get only the valid audio
    const validTarWaves = tarWaves.slice([0, 0, validStart], [1, 2, mixLength]);
    tarWaves.dispose();

    // Extract the source
    const source = validTarWaves;
    this.debug(`Final source shape: ${source.shape}`);

    const end = performance.now();
    console.log(`demix took ${(end - start).toFixed(2)}ms`);

    // Apply compensation if not matching the mix
    if (!isMatchMix) {
      console.debug(`Applying compensation factor: ${this.compensate}`);
      const compensatedSource = source.mul(this.compensate);
      source.dispose();
      console.debug('Demixing process completed.');
      return compensatedSource;
    } else {
      console.debug('Demixing process completed.');
      return source;
    }
  }

  /**
   * Apply window to tarWaves and update result and divider tensors
   * Handles both windowed and non-windowed cases
   */
  private applyWindow(
    tarWaves: tf.Tensor,
    window: tf.Tensor | null,
    result: tf.Tensor,
    divider: tf.Tensor,
    start: number,
    end: number,
    chunkSizeActual: number
  ): { result: tf.Tensor, divider: tf.Tensor } {
    // Calculate actual slice size
    const sliceSize = end - start;

    if (window !== null) {
      // Apply window to the tar_waves
      console.log("window mul")
      const windowedTarWaves = tf.mul(tarWaves.slice([0, 0, 0], [1, 2, chunkSizeActual]), window);
      console.log("window mul done")

      // Update divider with window
      const newDivider = this.updateTensorSlice(divider, window, start, end);

      // Update result with windowed tar_waves
      const newResult = this.updateTensorSlice(result, windowedTarWaves, start, end);

      // Clean up
      windowedTarWaves.dispose();

      return { result: newResult, divider: newDivider };
    } else {
      // Create a ones tensor for divider update
      const ones = tf.ones([1, 2, sliceSize]);

      // Update divider with ones
      const newDivider = this.updateTensorSlice(divider, ones, start, end);

      // Update result with tar_waves slice
      const slicedTarWaves = tarWaves.slice([0, 0, 0], [1, 2, sliceSize]);
      const newResult = this.updateTensorSlice(result, slicedTarWaves, start, end);

      // Clean up
      ones.dispose();
      slicedTarWaves.dispose();

      return { result: newResult, divider: newDivider };
    }
  }

  /**
 * Helper function to update a slice of a tensor with new values
 * Equivalent to: tensor[..., start:end] += values
 */
  private updateTensorSlice(tensor: tf.Tensor, values: tf.Tensor, start: number, end: number): tf.Tensor {
    // Ensure the values tensor has the right size for the slice
    const sliceSize = end - start;
    console.log(`Slicing ${tensor.shape}`)
    const slicedValues = values.slice([0, 0, 0], [1, 2, sliceSize]);

    // Create padded version of values to match full tensor shape
    const padded = tf.pad(
      slicedValues,
      [
        [0, 0],                         // No padding on batch dimension
        [0, 0],                         // No padding on channel dimension
        [start, tensor.shape[2] - end]  // Pad before and after to match tensor shape
      ]
    );

    // Add padded values to original tensor
    const updated = tensor.add(padded);

    // Clean up intermediates
    slicedValues.dispose();
    padded.dispose();

    return updated;
  }
  /**
   * Process the input mix through the model to separate sources
   * @param mix Input tensor of shape [1, 2, chunkSize]
   * @param isMatchMix Whether to match the mix directly or apply model
   * @returns Processed audio tensor of shape [1, 2, chunkSize]
   */
  private async runModel(mix: tf.Tensor, isMatchMix: boolean = false): Promise<tf.Tensor> {
    // Apply STFT to the mix
    this.debugTensorStats(mix, `runModel input (isMatchMix=${isMatchMix})`);

    const spek = this.stft.forward(mix);
    this.debugTensorStats(spek, "STFT output");

    // Zero out the first 3 bins of the spectrum to reduce low-frequency noise
    // We'll use tf.tidy to manage memory for these operations
    const spekZeroed = tf.tidy(() => {
      // Get shape for masking
      const [batchSize, channels, freqBins, timeFrames] = spek.shape;

      // Create mask with zeros for first 3 bins
      const freqMask = tf.concat([
        tf.zeros([3]),                  // First 3 bins are zero
        tf.ones([freqBins - 3])         // Rest are ones
      ], 0);

      // Reshape mask and tile to match spek dimensions
      const reshapedMask = freqMask.reshape([1, 1, freqBins, 1])
        .tile([batchSize, channels, 1, timeFrames]);

      // Apply mask to zero out bins
      return tf.mul(spek, reshapedMask);
    });

    let specPred: tf.Tensor;

    if (isMatchMix) {
      // In match_mix mode, use the STFT output directly
      specPred = spekZeroed;
      this.debug("isMatchMix: spectrum prediction obtained directly from STFT output.");
    } else {
      if (this.enableDenoise) {
        // Denoising not implemented yet
        throw new Error("Denoising mode is not yet implemented");
      } else {
        // Standard model run
        const inputTensor = await this.tensorToOnnxFormat(spekZeroed);
        const feeds = { input: new ort.Tensor('float32', inputTensor.data as Float32Array, inputTensor.dims) };
        const results = await this.model!.run(feeds);
        const output = results.output.data as Float32Array;

        // Convert output back to TF tensor format
        specPred = this.onnxOutputToTensor(output, spekZeroed.shape);
        this.debug("Model run on the spectrum without denoising.");
      }
    }

    // Apply inverse STFT to convert back to time domain
    const result = this.stft.inverse(specPred);
    this.debugTensorStats(result, `runModel output (isMatchMix=${isMatchMix})`);

    // Clean up tensors
    spek.dispose();
    if (specPred !== spekZeroed) {
      specPred.dispose();
    }
    spekZeroed.dispose();

    return result;
  }

  /**
   * Convert TensorFlow.js tensor to ONNX format
   */
  private async tensorToOnnxFormat(tensor: tf.Tensor): Promise<ort.Tensor> {
    // Get tensor shape for ONNX
    const shape = tensor.shape;

    // Convert to typed array (Float32Array)
    const data = await tensor.data() as Float32Array;

    // Create and return ONNX tensor with same shape
    return new ort.Tensor('float32', data, shape);
  }

  /**
   * Convert ONNX output back to TensorFlow.js tensor
   */
  private onnxOutputToTensor(output: Float32Array, shape: number[]): tf.Tensor {
    // Create tensor with the same shape as the original
    return tf.tensor(Array.from(output), shape);
  }
  /**
   * Subtract one set of channels from another using tensor operations
   * @param mix First tensor of shape [batch, channels, time] or [channels, time]
   * @param source Second tensor of shape [batch, channels, time] or [channels, time]
   * @returns Tensor with mix - source
   */
  private subtractChannels(mix: tf.Tensor, source: tf.Tensor): tf.Tensor {
    console.debug(`Subtracting channels: mix shape=${mix.shape}, source shape=${source.shape}`);

    return tf.tidy(() => {
      // Handle dimension mismatches
      let mixTensor = mix;
      let sourceTensor = source;

      // 1. First ensure both have same number of dimensions
      // If mix is 2D [channels, time] and source is 3D [batch, channels, time]
      if (mix.shape.length === 2 && source.shape.length === 3) {
        console.debug('Adding batch dimension to mix tensor');
        mixTensor = mix.expandDims(0);
      }
      // If source is 2D [channels, time] and mix is 3D [batch, channels, time]
      else if (source.shape.length === 2 && mix.shape.length === 3) {
        console.debug('Adding batch dimension to source tensor');
        sourceTensor = source.expandDims(0);
      }

      console.debug(`After dimension adjustment: mix=${mixTensor.shape}, source=${sourceTensor.shape}`);

      // 2. Now handle time dimension mismatch
      const mixTimeLength = mixTensor.shape[mixTensor.shape.length - 1];
      const sourceTimeLength = sourceTensor.shape[sourceTensor.shape.length - 1];

      if (mixTimeLength !== sourceTimeLength) {
        console.debug(`Time length mismatch: mix=${mixTimeLength}, source=${sourceTimeLength}`);

        // Instead of truncating, let's pad the shorter one to match the longer one
        // This preserves more audio data
        if (mixTimeLength > sourceTimeLength) {
          // Pad source to match mix length
          const padSize = mixTimeLength - sourceTimeLength;
          console.debug(`Padding source with ${padSize} zeros`);

          const paddings = Array(sourceTensor.shape.length).fill([0, 0]);
          paddings[paddings.length - 1] = [0, padSize]; // Pad only the time dimension

          sourceTensor = tf.pad(sourceTensor, paddings);
        } else {
          // Pad mix to match source length
          const padSize = sourceTimeLength - mixTimeLength;
          console.debug(`Padding mix with ${padSize} zeros`);

          const paddings = Array(mixTensor.shape.length).fill([0, 0]);
          paddings[paddings.length - 1] = [0, padSize]; // Pad only the time dimension

          mixTensor = tf.pad(mixTensor, paddings);
        }

        console.debug(`After padding: mix=${mixTensor.shape}, source=${sourceTensor.shape}`);
      }

      // 3. Perform subtraction with tensors of matching dimensions
      const result = tf.sub(mixTensor, sourceTensor);
      console.debug(`Subtraction result shape: ${result.shape}`);

      return result;
    });
  }
  /**
   * Save audio output to a file (in browser, creates a blob URL)
   * @param outputPath The path where the audio would be saved (used as identifier)
   * @param audioData Tensor or Float32Array[] containing audio data
   * @param stemName Name of the stem for logging purposes
   * @returns URL for the blob containing the WAV file
   */
  private async saveAudioOutput(outputPath: string, audioData: tf.Tensor | Float32Array[], stemName: string): Promise<string> {
    // Since we're running in a browser, we can't actually save to the filesystem
    // Instead, we'll create a blob and return a downloadable URL
    this.debug(`Saving ${stemName} to ${outputPath}`);

    try {
      // Convert tensor to Float32Array[] if needed
      let audioChannels: Float32Array[];

      if (audioData instanceof tf.Tensor) {
        this.debug(`Converting tensor with shape ${audioData.shape} to Float32Arrays`);
        audioChannels = await this.tensorToArrays(audioData);
        audioData.dispose();
      } else {
        audioChannels = audioData;
      }

      this.debug(`Audio data has ${audioChannels.length} channels with ${audioChannels[0]?.length || 0} samples each`);

      // Create a WAV blob
      const blob = await AudioUtils.saveAudioFile(audioChannels, this.sampleRate, outputPath);

      // Create a blob URL
      const blobUrl = URL.createObjectURL(blob);

      this.info(`Created ${stemName} output blob: ${blob.size} bytes`);

      // Store the blob and URL for later access
      if (typeof window !== 'undefined') {
        (window as any).outputBlobs = (window as any).outputBlobs || {};
        (window as any).outputBlobs[outputPath] = {
          url: blobUrl,
          blob: blob,
          name: stemName,
          size: blob.size
        };
      }

      return blobUrl;

    } catch (error) {
      this.error(`Error saving ${stemName} output: ${error}`);
      throw error;
    }
  }

  /**
   * Convert a tensor to array of Float32Arrays for audio processing
   * Handles tensors of shape [batch, channels, samples] or [channels, samples]
   */
  private async tensorToArrays(tensor: tf.Tensor): Promise<Float32Array[]> {
    const shape = tensor.shape;
    this.debug(`Converting tensor with shape ${shape} to arrays`);

    // Extract tensor data based on its shape
    let channelsData: number[][];

    if (shape.length === 3) {
      // [batch, channels, samples] - extract first batch
      const batchData = await tensor.slice([0, 0, 0], [1, -1, -1]).squeeze([0]).array() as number[][];
      channelsData = batchData;
    } else if (shape.length === 2) {
      // [channels, samples]
      channelsData = await tensor.array() as number[][];
    } else {
      throw new Error(`Unsupported tensor shape for audio: ${shape}`);
    }

    // Convert to Float32Array
    const result: Float32Array[] = [];
    for (let i = 0; i < channelsData.length; i++) {
      result.push(new Float32Array(channelsData[i]));
    }

    this.debug(`Converted tensor to ${result.length} channels`);
    return result;
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
export { MDXArchConfig };
