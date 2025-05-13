import * as ort from 'onnxruntime-web';
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
  private primarySource?: Float32Array[];
  private secondarySource?: Float32Array[];
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
    this.chunkSize = (this.segmentSize - 1) * this.hopLength + this.nFft;

    // gen_size is the chunk size minus twice the trim size
    this.genSize = this.chunkSize - 2 * this.trim;

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
    const audioChannels = await AudioUtils.loadAudioFile(audioFilePath, this.sampleRate);

    // Prepare the mix for processing
    this.debug(`Preparing mix for input audio file ${this.audioFilePath}...`);
    const mix = await this.prepareMix(audioChannels);

    this.debug('Normalizing mix before demixing...');
    const normalizedMix = this.normalize(mix, this.normalizationThreshold, this.amplificationThreshold);

    // Start the demixing process
    const source = await this.demix(normalizedMix);
    this.debug('Demixing completed.');

    // Initialize the list for output files
    const outputFiles: string[] = [];
    this.debug('Processing output files...');

    // TODO: Implement the rest of the separation logic
    // For now, just return empty array
    return outputFiles;
  }

  // Method to prepare mix for processing
  async prepareMix(audioData: Float32Array | Float32Array[]): Promise<Float32Array[]> {
    this.debug(`Preparing mix for input audio...`);
    return super.prepareMix(audioData);
  }

  // Method to normalize audio
  normalize(wave: Float32Array | Float32Array[], maxPeak: number, minPeak: number): Float32Array[] {
    this.debug('Normalizing audio...');
    return super.normalize(wave, maxPeak, minPeak);
  }

  private async demix(mix: Float32Array[], isMatchMix: boolean = false): Promise<Float32Array[]> {
    this.debug(`Starting demixing process with isMatchMix: ${isMatchMix}...`);
    this.initializeModelSettings();

    // Preserve the original mix for later use
    const orgMix = mix;
    this.debug(`Original mix stored. Shape: ${mix.length}x${mix[0].length}`);

    // Initialize mix for processing
    const [mixWaves, pad] = this.initializeMix(mix);

    // Process chunks
    const processedChunks: Float32Array[][] = [];

    // Process each chunk (stereo pairs)
    for (let i = 0; i < mixWaves.length; i += 2) {
      const chunk = [mixWaves[i], mixWaves[i + 1]];
      const processed = await this.runModel(chunk, isMatchMix);
      processedChunks.push(processed);
    }

    // Combine processed chunks back into a single audio signal
    const combinedChannels: Float32Array[] = [
      new Float32Array(mix[0].length),
      new Float32Array(mix[1].length)
    ];

    let currentSample = 0;
    for (const chunk of processedChunks) {
      for (let ch = 0; ch < 2; ch++) {
        for (let s = 0; s < chunk[ch].length && currentSample + s < mix[ch].length; s++) {
          combinedChannels[ch][currentSample + s] = chunk[ch][s];
        }
      }
      currentSample += this.genSize;
    }

    // Apply compensation if not matching mix
    if (!isMatchMix) {
      for (let ch = 0; ch < combinedChannels.length; ch++) {
        for (let s = 0; s < combinedChannels[ch].length; s++) {
          combinedChannels[ch][s] *= this.compensate;
        }
      }
      this.debug('Match mix mode; compensate multiplier applied.');
    }

    this.debug('Demixing process completed.');
    return combinedChannels;
  }

  /**
   * Run the model on a chunk of audio
   */
  private async runModel(mixChunk: Float32Array[], isMatchMix: boolean = false): Promise<Float32Array[]> {
    if (!this.model || !this.stft) {
      throw new Error('Model or STFT not initialized');
    }

    // Apply STFT to the mix chunk
    this.debug(`Running STFT on the mix. Mix shape: ${mixChunk.length}x${mixChunk[0].length}`);
    const spek = this.stft.forward(mixChunk, this.segmentSize);
    this.debug(`STFT applied on mix. Spectrum shape: ${spek.length}x${spek[0].length}x${spek[0][0].length}x${spek[0][0][0].length}`);

    // Zero out the first 3 bins of the spectrum
    for (let ch = 0; ch < spek.length; ch++) {
      for (let bin = 0; bin < 3; bin++) {
        for (let frame = 0; frame < spek[ch][bin].length; frame++) {
          spek[ch][bin][frame][0] = 0; // Real part
          spek[ch][bin][frame][1] = 0; // Imaginary part
        }
      }
    }

    let specPred: number[][][][] | Float32Array[][][][] = spek;

    if (isMatchMix) {
      this.debug('isMatchMix: spectrum prediction obtained directly from STFT output.');
    } else {
      // Prepare input tensor for ONNX model
      // Convert complex spectrogram to real tensor format expected by the model
      const batchSize = 1; // Processing one stereo pair at a time
      const frames = spek[0][0].length;
      this.debug(`Preparing model input. Frames: ${frames}, Expected: ${this.segmentSize}`);

      if (frames !== this.segmentSize) {
        this.debug(`Frame count mismatch: got ${frames}, expected ${this.segmentSize}`);
      }

      const inputTensor = this.prepareModelInput(spek);

      if (this.enableDenoise) {
        // Run model on negative spectrum
        const negInputTensor = inputTensor.map(val => -val);
        const negFeeds = { input: new ort.Tensor('float32', negInputTensor, [batchSize, 4, this.dimF, frames]) };
        const negResults = await this.model.run(negFeeds);
        const negOutput = negResults.output.data as Float32Array;

        // Run model on positive spectrum
        const posFeeds = { input: new ort.Tensor('float32', inputTensor, [batchSize, 4, this.dimF, frames]) };
        const posResults = await this.model.run(posFeeds);
        const posOutput = posResults.output.data as Float32Array;

        // Combine results
        const combinedOutput = new Float32Array(posOutput.length);
        for (let i = 0; i < combinedOutput.length; i++) {
          combinedOutput[i] = (negOutput[i] * -0.5) + (posOutput[i] * 0.5);
        }

        specPred = this.reshapeModelOutput(combinedOutput, spek);
        this.debug('Model run on both negative and positive spectrums for denoising.');
      } else {
        const feeds = { input: new ort.Tensor('float32', inputTensor, [batchSize, 4, this.dimF, frames]) };
        this.debug(`Creating tensor with shape: [${batchSize}, 4, ${this.dimF}, ${frames}]`);
        const results = await this.model.run(feeds);
        const output = results.output.data as Float32Array;
        specPred = this.reshapeModelOutput(output, spek);
        this.debug('Model run on the spectrum without denoising.');
      }
    }

    // Apply inverse STFT to convert back to time domain
    const result = this.stft.inverse(specPred);
    this.debug(`Inverse STFT applied. Returning result with shape: ${result.length}x${result[0].length}`);

    return result;
  }

  /**
   * Prepare model input from complex spectrogram
   */
  private prepareModelInput(spek: Float32Array[][][][]): Float32Array {
    const channels = spek.length;
    const freqBins = spek[0].length;
    const frames = spek[0][0].length;

    // Model expects real and imaginary parts as separate channels
    // Input shape: [batch, channels * 2, freq_bins, frames]
    const outputSize = channels * 2 * freqBins * frames;
    const output = new Float32Array(outputSize);

    let idx = 0;
    // Real parts
    for (let ch = 0; ch < channels; ch++) {
      for (let f = 0; f < freqBins; f++) {
        for (let t = 0; t < frames; t++) {
          output[idx++] = spek[ch][f][t][0];
        }
      }
    }
    // Imaginary parts
    for (let ch = 0; ch < channels; ch++) {
      for (let f = 0; f < freqBins; f++) {
        for (let t = 0; t < frames; t++) {
          output[idx++] = spek[ch][f][t][1];
        }
      }
    }

    return output;
  }

  /**
   * Reshape model output back to complex spectrogram format
   */
  private reshapeModelOutput(output: Float32Array, originalSpek: Float32Array[][][][]): number[][][][] {
    const channels = originalSpek.length;
    const freqBins = originalSpek[0].length;
    const frames = originalSpek[0][0].length;

    const result: number[][][][] = [];

    for (let ch = 0; ch < channels; ch++) {
      const channelData: number[][][] = [];
      for (let f = 0; f < freqBins; f++) {
        const freqData: number[][] = [];
        for (let t = 0; t < frames; t++) {
          freqData.push([0, 0]);
        }
        channelData.push(freqData);
      }
      result.push(channelData);
    }

    let idx = 0;
    // Real parts
    for (let ch = 0; ch < channels; ch++) {
      for (let f = 0; f < freqBins; f++) {
        for (let t = 0; t < frames; t++) {
          result[ch][f][t][0] = output[idx++];
        }
      }
    }
    // Imaginary parts
    for (let ch = 0; ch < channels; ch++) {
      for (let f = 0; f < freqBins; f++) {
        for (let t = 0; t < frames; t++) {
          result[ch][f][t][1] = output[idx++];
        }
      }
    }

    return result;
  }

  private initializeMix(mix: Float32Array[], isCheckpoint: boolean = false): [Float32Array[], number] {
    this.debug(`Initializing mix with isCheckpoint=${isCheckpoint}. Initial mix shape: ${mix.length}x${mix[0].length}`);

    // Ensure the mix is a 2-channel (stereo) audio signal
    if (mix.length !== 2) {
      const errorMessage = `Expected a 2-channel audio signal, but got ${mix.length} channels`;
      this.error(errorMessage);
      throw new Error(errorMessage);
    }

    let mixWaves: Float32Array[] = [];
    let pad = 0;

    if (isCheckpoint) {
      this.debug('Processing in checkpoint mode...');
      // Calculate padding based on the generation size and trim
      pad = this.genSize + this.trim - (mix[0].length % this.genSize);
      this.debug(`Padding calculated: ${pad}`);

      // Add padding at the beginning and the end of the mix
      const paddedMix = mix.map(channel => {
        const padded = new Float32Array(this.trim + channel.length + pad);
        padded.set(channel, this.trim);
        return padded;
      });

      // Determine the number of chunks based on the mixture's length
      const numChunks = Math.floor(paddedMix[0].length / this.genSize);
      this.debug(`Mixture shape after padding: ${paddedMix[0].length}, Number of chunks: ${numChunks}`);

      // Split the mixture into chunks
      for (let i = 0; i < numChunks; i++) {
        const chunk = paddedMix.map(channel =>
          channel.slice(i * this.genSize, i * this.genSize + this.chunkSize)
        );
        mixWaves.push(...chunk);
      }
    } else {
      // If not in checkpoint mode, process normally
      this.debug('Processing in non-checkpoint mode...');
      const nSample = mix[0].length;

      // Calculate necessary padding to make the total length divisible by the generation size
      pad = this.genSize - nSample % this.genSize;
      this.debug(`Number of samples: ${nSample}, Padding calculated: ${pad}`);

      // Apply padding to the mix
      const mixP = mix.map(channel => {
        const padded = new Float32Array(this.trim + channel.length + pad + this.trim);
        padded.set(channel, this.trim);
        return padded;
      });
      this.debug(`Shape of mix after padding: ${mixP[0].length}`);

      // Process the mix in chunks
      let i = 0;
      while (i < nSample + pad) {
        // Create a chunk that will produce exactly segmentSize frames
        const chunkStart = i;
        const chunkEnd = i + this.chunkSize;

        const waves = mixP.map(channel => {
          const chunk = new Float32Array(this.chunkSize);
          const sourceEnd = Math.min(chunkEnd, channel.length);
          const copyLength = sourceEnd - chunkStart;
          if (copyLength > 0) {
            chunk.set(channel.slice(chunkStart, sourceEnd));
          }
          return chunk;
        });

        mixWaves.push(...waves);
        this.debug(`Processed chunk ${mixWaves.length}: Start ${i}, End ${i + this.chunkSize}`);
        i += this.genSize;
      }
    }

    this.debug(`Converted mix to chunks. Total chunks: ${mixWaves.length}`);
    return [mixWaves, pad];
  }
}
