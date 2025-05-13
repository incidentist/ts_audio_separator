import { test, expect } from '@playwright/test';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

test.describe('MDX Separator Integration Tests', () => {
  test.beforeAll(async () => {
    // Ensure fixtures directory exists
    const fixturesDir = join(__dirname, 'fixtures');
    mkdirSync(fixturesDir, { recursive: true });

    // Create a simple test WAV file
    const sampleRate = 44100;
    const duration = 1; // 1 second
    const samples = sampleRate * duration;
    const audioData = new Float32Array(samples);

    // Generate a simple sine wave
    for (let i = 0; i < samples; i++) {
      audioData[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
    }

    // Create WAV file
    const createWAVBuffer = (audioData: Float32Array, sampleRate: number): Buffer => {
      const length = audioData.length;
      const arrayBuffer = new ArrayBuffer(44 + length * 2);
      const view = new DataView(arrayBuffer);

      // WAV header
      const writeString = (offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };

      writeString(0, 'RIFF');
      view.setUint32(4, 36 + length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true); // mono
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, length * 2, true);

      // Convert float to 16-bit PCM
      for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(44 + i * 2, Math.floor(sample * 32767), true);
      }

      return Buffer.from(arrayBuffer);
    };

    const wavBuffer = createWAVBuffer(audioData, sampleRate);
    const testFilePath = join(__dirname, 'fixtures', 'test-audio.wav');
    writeFileSync(testFilePath, wavBuffer);
  });

  test('should load model, prepare mix, normalize, and process audio file', async ({ page }) => {
    page.on('console', msg => console.log('Browser console:', msg.text()));
    page.on('pageerror', error => console.log('Page error:', error));
    // Log all network requests to see what's being loaded
    page.on('request', request => console.log('Request:', request.url()));
    page.on('response', response => console.log('Response:', response.url(), response.status()));

    // Serve the test file
    await page.route('**/test-audio.wav', async route => {
      const path = join(__dirname, 'fixtures', 'test-audio.wav');
      await route.fulfill({ path });
    });

    await page.goto('http://localhost:5173/test-page.html');


    // Execute the test in the browser context
    const result = await page.evaluate(async () => {
      // Wait for the library to be available
      console.log("one");
      const checkLibrary = () => new Promise<void>((resolve) => {
        console.log("two");
        const interval = setInterval(() => {
          if ((window as any).WebDemix2) {
            clearInterval(interval);
            resolve();
          }
        }, 100);
      });

      await checkLibrary();
      console.log("three");
      // @ts-ignore - This will be available in the browser context
      const { MDXSeparator } = window.WebDemix2;

      // Configure the separator
      const separator = new MDXSeparator(
        {
          modelPath: 'UVR_MDXNET_KARA_2.onnx',
          modelName: 'UVR_MDXNET_KARA_2',
          modelData: {
            compensate: 1.0,
            mdx_dim_f_set: 3072,
            mdx_dim_t_set: 8,
            mdx_n_fft_scale_set: 7680
          },
          logLevel: 'debug'
        },
        {
          segmentSize: 256,
          overlap: 0.25,
          batchSize: 1,
          hopLength: 1024,
          enableDenoise: false
        }
      );

      // Load the model (this will download it)
      await separator.loadModel();

      // Test with audio file
      const audioUrl = '/test-audio.wav';

      // Test separate method (which includes prepareMix and normalize)
      let separateError = null;
      try {
        await separator.separate(audioUrl);
      } catch (error: any) {
        separateError = error.message;
      }

      // Also test individual methods
      const testData = new Float32Array(44100); // 1 second of silence
      for (let i = 0; i < testData.length; i++) {
        testData[i] = Math.sin(2 * Math.PI * 440 * i / 44100) * 0.3;
      }

      const prepared = await separator.prepareMix(testData);
      const normalized = separator.normalize(prepared, 0.9, 0.6);

      return {
        modelLoaded: true,
        preparedChannels: prepared.length,
        normalizedChannels: normalized.length,
        separateError,
        normalizedMax: Math.max(...normalized[0])
      };
    });

    expect(result.modelLoaded).toBe(true);
    expect(result.preparedChannels).toBe(2);
    expect(result.normalizedChannels).toBe(2);
    expect(result.normalizedMax).toBeLessThanOrEqual(0.9);

    // Separate method should work without throwing
    expect(result.separateError).toBeNull();
  });
});
