// Base class for all separator architectures
export abstract class CommonSeparator {
  protected outputDir: string;
  protected outputFormat: string;
  protected normalizationThreshold: number;
  protected amplificationThreshold: number;
  protected logLevel: string;
  protected sampleRate: number;

  // Model-specific parameters
  protected modelPath: string;
  protected modelData: any;
  protected modelName: string;
  protected primaryStemName: string;
  protected secondaryStemName: string;
  protected outputSingleStem?: string;
  protected invertUsingSpec: boolean;

  constructor(config: SeparatorConfig) {
    this.outputDir = config.outputDir || './output';
    this.outputFormat = config.outputFormat || 'wav';
    this.normalizationThreshold = config.normalizationThreshold ?? 0.9;
    this.amplificationThreshold = config.amplificationThreshold ?? 0.6;
    this.logLevel = config.logLevel || 'info';
    this.sampleRate = config.sampleRate || 44100;

    this.modelPath = config.modelPath;
    this.modelData = config.modelData || {};
    this.modelName = config.modelName;
    this.primaryStemName = config.primaryStemName || 'vocals';
    this.secondaryStemName = config.secondaryStemName || 'instrumental';
    this.outputSingleStem = config.outputSingleStem;
    this.invertUsingSpec = config.invertUsingSpec ?? false;
  }

  // Abstract methods that subclasses must implement
  abstract loadModel(): Promise<void>;
  abstract separate(audioFilePath: string, customOutputNames?: Record<string, string>): Promise<string[]>;

  // Common utility methods
  protected log(level: string, message: string): void {
    if (this.shouldLog(level)) {
      console.log(`[${level.toUpperCase()}] ${message}`);
    }
  }

  protected debug(message: string): void {
    this.log('debug', message);
  }

  protected info(message: string): void {
    this.log('info', message);
  }

  protected warning(message: string): void {
    this.log('warning', message);
  }

  protected error(message: string): void {
    this.log('error', message);
  }

  private shouldLog(level: string): boolean {
    const levels = ['error', 'warning', 'info', 'debug'];
    const configLevel = levels.indexOf(this.logLevel);
    const messageLevel = levels.indexOf(level);
    return messageLevel <= configLevel;
  }

  // Utility method to get stem output path
  protected getStemOutputPath(stemName: string, customOutputNames?: Record<string, string>): string {
    if (customOutputNames && customOutputNames[stemName]) {
      return customOutputNames[stemName];
    }
    return `${this.outputDir}/${this.modelName}_${stemName}.${this.outputFormat}`;
  }

  // Common audio processing methods
  protected async prepareMix(audioData: Float32Array | Float32Array[]): Promise<Float32Array[]> {
    // If input is already 2D array, return it
    if (Array.isArray(audioData)) {
      return audioData;
    }

    // Convert mono to stereo by duplicating the channel
    if (audioData instanceof Float32Array) {
      return [audioData, audioData];
    }

    throw new Error('Invalid audio data format');
  }

  protected normalize(wave: Float32Array | Float32Array[], maxPeak: number, minPeak: number): Float32Array[] {
    // Convert to array if single channel
    const channels = Array.isArray(wave) ? wave : [wave];

    // Find the maximum absolute value across all channels
    let maxVal = 0;
    for (const channel of channels) {
      for (let i = 0; i < channel.length; i++) {
        maxVal = Math.max(maxVal, Math.abs(channel[i]));
      }
    }

    // Calculate normalization factor
    let normFactor = 1;
    if (maxVal > maxPeak) {
      normFactor = maxPeak / maxVal;
    } else if (maxVal < minPeak && maxVal > 0) {
      normFactor = minPeak / maxVal;
    }

    // Apply normalization
    return channels.map(channel => {
      const normalized = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) {
        normalized[i] = channel[i] * normFactor;
      }
      return normalized;
    });
  }
}

export interface SeparatorConfig {
  outputDir?: string;
  outputFormat?: string;
  normalizationThreshold?: number;
  amplificationThreshold?: number;
  logLevel?: string;
  sampleRate?: number;
  modelPath: string;
  modelData?: any;
  modelName: string;
  primaryStemName?: string;
  secondaryStemName?: string;
  outputSingleStem?: string;
  invertUsingSpec?: boolean;
}
