import * as tf from '@tensorflow/tfjs';

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

  /** Converts mono → stereo and passes through existing stereo. */
  protected async prepareMix(audioData: tf.Tensor): Promise<tf.Tensor2D> {
    return tf.tidy(() => {
      if (audioData.rank === 2) {
        // Already [channels, samples] – return as-is.
        return audioData as tf.Tensor2D;
      }
      if (audioData.rank === 1) {
        // Duplicate the mono channel → shape [2, samples].
        return tf.stack([audioData, audioData]) as tf.Tensor2D;
      }
      throw new Error('Invalid audio data format (expect 1-D or 2-D Tensor)');
    });
  }
  protected async normalize(
    wave: tf.Tensor,            // tf.Tensor1D | tf.Tensor2D
    maxPeak = 1.0,
    minPeak?: number            // optional
  ): Promise<tf.Tensor> {       // returns same rank as input
    const mono = wave.rank === 1;
    const audio = mono ? wave : wave as tf.Tensor2D;

    const [peak] = await tf.max(tf.abs(audio)).data();
    let gain = 1;
    if (peak > maxPeak) {
      gain = maxPeak / peak;
    } else if (minPeak !== undefined && peak < minPeak && peak > 0) {
      gain = minPeak / peak;
    }

    const out = tf.tidy(() => audio.mul(gain));
    return mono ? out.squeeze([0]) : out;   // keep original rank
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
