import * as ort from 'onnxruntime-web';
import { CommonSeparator, SeparatorConfig } from '../common_separator';
import { AudioUtils } from '../../utils/audio_utils';

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

    // chunk_size is the hop_length size times the segment size minus one
    this.chunkSize = this.hopLength * (this.segmentSize - 1);

    // gen_size is the chunk size minus twice the trim size
    this.genSize = this.chunkSize - 2 * this.trim;

    this.debug(`Model input params: nFft=${this.nFft} hopLength=${this.hopLength} dimF=${this.dimF}`);
    this.debug(`Model settings: nBins=${this.nBins}, trim=${this.trim}, chunkSize=${this.chunkSize}, genSize=${this.genSize}`);
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

    // TODO: Implement the rest of the demixing logic
    this.debug('Demixing process completed.');

    // For now, return the original mix
    return mix;
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
        const waves = mixP.map(channel =>
          channel.slice(i, i + this.chunkSize)
        );
        mixWaves.push(...waves);
        this.debug(`Processed chunk ${mixWaves.length}: Start ${i}, End ${i + this.chunkSize}`);
        i += this.genSize;
      }
    }

    this.debug(`Converted mix to chunks. Total chunks: ${mixWaves.length}`);
    return [mixWaves, pad];
  }
}
