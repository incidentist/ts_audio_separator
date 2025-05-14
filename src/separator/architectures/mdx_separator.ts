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

    // Normalize and process the primary source if it's not already an array
    if (!this.primarySource) {
      this.debug('Normalizing primary source...');
      this.primarySource = this.normalize(source, this.normalizationThreshold, this.amplificationThreshold);
    }

    // Process the secondary source if not already an array
    if (!this.secondarySource) {
      this.debug('Producing secondary source: demixing in match_mix mode');
      const rawMix = await this.demix(normalizedMix, true);

      if (this.invertUsingSpec) {
        this.debug('Inverting secondary stem using spectogram as invertUsingSpec is set to true');
        // For spectral inversion, we'd need to implement this method
        // For now, use simple subtraction
        this.secondarySource = this.subtractChannels(normalizedMix, source);
      } else {
        this.debug('Inverting secondary stem by subtracting demixed stem from original mix');
        this.secondarySource = this.subtractChannels(normalizedMix, source);
      }
    }

    // Save and process the secondary stem if needed
    if (!this.outputSingleStem || this.outputSingleStem.toLowerCase() === this.secondaryStemName.toLowerCase()) {
      const secondaryPath = this.getStemOutputPath(this.secondaryStemName, customOutputNames);

      this.info(`Saving ${this.secondaryStemName} stem to ${secondaryPath}...`);
      const secondaryUrl = await this.saveAudioOutput(secondaryPath, this.secondarySource!, this.secondaryStemName);
      outputFiles.push(secondaryUrl);
    }

    // Save and process the primary stem if needed
    if (!this.outputSingleStem || this.outputSingleStem.toLowerCase() === this.primaryStemName.toLowerCase()) {
      const primaryPath = this.getStemOutputPath(this.primaryStemName, customOutputNames);

      if (!this.primarySource) {
        this.primarySource = source;
      }

      this.info(`Saving ${this.primaryStemName} stem to ${primaryPath}...`);
      const primaryUrl = await this.saveAudioOutput(primaryPath, this.primarySource, this.primaryStemName);
      outputFiles.push(primaryUrl);
    }

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

    // Preserves the original mix for later use
    const orgMix = mix;
    this.debug(`Original mix stored. Shape: ${mix.length}x${mix[0].length}`);

    // Initializes a list to store the separated waveforms
    const tarWaves_: Float32Array[][] = [];

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
    const pad = genSize + this.trim - ((mix[0].length) % genSize);

    // Prepare the mixture with padding at the beginning and the end
    const mixture = mix.map(channel => {
      const padded = new Float32Array(this.trim + channel.length + pad);
      padded.set(channel, this.trim);
      return padded;
    });
    this.debug(`Mixture prepared with padding. Mixture shape: ${mixture[0].length}`);

    // Calculate the step size for processing chunks based on the overlap
    const step = Math.floor((1 - overlap) * chunkSize);
    this.debug(`Step size for processing chunks: ${step} as overlap is set to ${overlap}.`);

    // Initialize arrays to store the results and to account for overlap
    const result = [
      new Float32Array(mixture[0].length),
      new Float32Array(mixture[1].length)
    ];
    const divider = [
      new Float32Array(mixture[0].length),
      new Float32Array(mixture[1].length)
    ];

    // Initialize counters for processing chunks
    let total = 0;
    const totalLength = mixture[0].length;
    const totalChunks = Math.ceil(totalLength / step);
    this.debug(`Total chunks to process: ${totalChunks}`);

    // Process each chunk of the mixture
    for (let i = 0; i < totalLength; i += step) {
      total++;
      const start = i;
      const end = Math.min(i + chunkSize, totalLength);
      this.debug(`Processing chunk ${total}/${totalChunks}: Start ${start}, End ${end}`);

      // Handle windowing for overlapping chunks
      const chunkSizeActual = end - start;
      let window: Float32Array | null = null;

      if (overlap !== 0) {
        window = new Float32Array(chunkSizeActual);
        for (let j = 0; j < chunkSizeActual; j++) {
          window[j] = 0.5 - 0.5 * Math.cos(2 * Math.PI * j / (chunkSizeActual - 1));
        }
        this.debug('Window applied to the chunk.');
      }

      // Zero-pad the chunk to prepare it for processing
      const mixPart = mixture.map(channel => {
        const chunk = channel.slice(start, end);
        if (end !== i + chunkSize) {
          const paddedChunk = new Float32Array(chunkSize);
          paddedChunk.set(chunk);
          return paddedChunk;
        }
        return chunk;
      });

      // Run the model to separate the sources
      const tarWaves = await this.runModel(mixPart, isMatchMix);

      // Apply windowing if needed and accumulate the results
      for (let ch = 0; ch < 2; ch++) {
        for (let s = 0; s < chunkSizeActual; s++) {
          if (window !== null) {
            tarWaves[ch][s] *= window[s];
            divider[ch][start + s] += window[s];
          } else {
            divider[ch][start + s] += 1;
          }
          result[ch][start + s] += tarWaves[ch][s];
        }
      }
    }

    // Normalize the results by the divider to account for overlap
    this.debug('Normalizing result by dividing result by divider.');
    const tarWaves = result.map((channel, ch) => {
      const normalized = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) {
        normalized[i] = divider[ch][i] !== 0 ? channel[i] / divider[ch][i] : 0;
      }
      return normalized;
    });

    tarWaves_.push(tarWaves);

    // Reshape the results to match the original dimensions
    const tarWavesFlat = tarWaves_.map(batch => {
      return batch.map(channel =>
        channel.slice(this.trim, this.trim + mix[0].length)
      );
    })[0]; // We only have one batch in this implementation

    // Extract the source from the results
    const source = tarWavesFlat;
    this.debug(`Concatenated tar_waves. Shape: ${source.length}x${source[0].length}`);

    // Apply compensation if not matching the mix
    if (!isMatchMix) {
      for (let ch = 0; ch < source.length; ch++) {
        for (let i = 0; i < source[ch].length; i++) {
          source[ch][i] *= this.compensate;
        }
      }
      this.debug('Compensate multiplier applied to non-match mix.');
    }

    this.debug('Demixing process completed.');
    return source;
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

  /**
   * Subtract one set of channels from another
   */
  private subtractChannels(mix: Float32Array[], source: Float32Array[]): Float32Array[] {
    const result: Float32Array[] = [];

    for (let ch = 0; ch < mix.length; ch++) {
      const channel = new Float32Array(mix[ch].length);
      for (let i = 0; i < channel.length; i++) {
        channel[i] = mix[ch][i] - source[ch][i];
      }
      result.push(channel);
    }

    return result;
  }

  /**
   * Save audio output to a file (in browser, creates a blob URL)
   */
  private async saveAudioOutput(outputPath: string, audioData: Float32Array[], stemName: string): Promise<string> {
    // Since we're running in a browser, we can't actually save to the filesystem
    // Instead, we'll create a blob and return a downloadable URL

    try {
      // Create a WAV blob
      const blob = await AudioUtils.saveAudioFile(audioData, this.sampleRate, outputPath);

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
}

export { MDXArchConfig };
