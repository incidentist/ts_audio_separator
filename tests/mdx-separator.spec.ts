import { test, expect } from '@playwright/test';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { existsSync, mkdirSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

test.describe('MDX Separator Integration Tests', () => {
  test('should process real audio file (heard_sound.wav)', async ({ page }) => {
    page.on('console', msg => console.log('Browser console:', msg.text()));
    page.on('pageerror', error => console.log('Page error:', error));

    // Log all network requests to see what's being loaded
    page.on('request', request => console.log('Request:', request.url()));
    page.on('response', response => console.log('Response:', response.url(), response.status()));

    // Check if the audio file exists
    const audioPath = join(__dirname, 'fixtures', 'heard_sound.wav');
    console.log('Checking for audio file at:', audioPath);
    expect(existsSync(audioPath)).toBe(true);

    // Serve the test file
    await page.route('**/heard_sound.wav', async route => {
      await route.fulfill({ path: audioPath });
    });

    await page.goto('http://localhost:5173/test-page.html');

    // Execute the test in the browser context
    const result = await page.evaluate(async () => {
      // Wait for the library to be available
      console.log("Waiting for WebDemix2...");
      const checkLibrary = () => new Promise<void>((resolve) => {
        const interval = setInterval(() => {
          if ((window as any).WebDemix2) {
            clearInterval(interval);
            resolve();
          }
        }, 100);
      });

      await checkLibrary();
      console.log("WebDemix2 loaded");

      const { MDXSeparator } = (window as any).WebDemix2;

      // Model configuration for UVR_MDXNET_KARA_2
      const modelData = {
        compensate: 1.065,
        mdx_dim_f_set: 2048,
        mdx_dim_t_set: 8,
        mdx_n_fft_scale_set: 5120
      };

      // Configure the separator
      const separator = new MDXSeparator(
        {
          modelPath: 'UVR_MDXNET_KARA_2.onnx',
          modelName: 'UVR_MDXNET_KARA_2',
          modelData: modelData,
          logLevel: 'debug',
          sampleRate: 44100,
          normalizationThreshold: 0.9,
          amplificationThreshold: 0.6,
          outputDir: 'output',
          primaryStemName: 'vocals',
          secondaryStemName: 'instrumental'
        },
        {
          segmentSize: 256,
          overlap: 0.25,
          batchSize: 1,
          hopLength: 1024,
          enableDenoise: false
        }
      );

      console.log("Loading model...");
      await separator.loadModel();
      console.log("Model loaded successfully");

      // Test with real audio file
      const audioUrl = '/heard_sound.wav';

      // Test separate method (which includes prepareMix and normalize)
      let separateResult = null;
      let separateError = null;

      try {
        console.log("Starting separation process...");
        separateResult = await separator.separate(audioUrl);
        console.log("Separation completed successfully");
      } catch (error: any) {
        console.error("Separation error:", error);
        separateError = error.message;
      }

      // Check if output blobs were created
      const outputBlobs = (window as any).outputBlobs || {};
      const blobUrls = Object.keys(outputBlobs);

      // Collect blob information without converting to base64
      const blobsInfo: any[] = [];
      for (const [path, data] of Object.entries(outputBlobs)) {
        const blobData = data as any;
        blobsInfo.push({
          path,
          name: blobData.name,
          size: blobData.size,
          url: blobData.url
        });
      }

      return {
        modelLoaded: true,
        separateError,
        separateResult,
        success: !separateError,
        outputBlobCount: blobUrls.length,
        outputBlobUrls: blobUrls,
        blobsInfo
      };
    });

    console.log('\n=== MDX Separator Real Audio Test Results ===\n');
    console.log('Model loaded:', result.modelLoaded);
    console.log('Separation successful:', result.success);
    console.log('Output files created:', result.outputBlobCount);

    if (result.separateError) {
      console.log('Error:', result.separateError);
    }

    if (result.separateResult) {
      console.log('Output URLs:', result.separateResult);
    }

    // Download and save the separated audio files
    if (result.blobsInfo && result.blobsInfo.length > 0) {
      const outputDir = join(__dirname, 'output');
      if (!existsSync(outputDir)) {
        mkdirSync(outputDir, { recursive: true });
      }

      for (const blobInfo of result.blobsInfo) {
        console.log(`Blob info: ${blobInfo.name} (${blobInfo.size} bytes) at ${blobInfo.url}`);

        // Trigger download programmatically and save to output directory
        const downloadPromise = page.waitForEvent('download');

        // Trigger download by creating a temporary link and clicking it
        await page.evaluate((url) => {
          const a = document.createElement('a');
          a.href = url;
          a.download = 'download.wav';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }, blobInfo.url);

        try {
          const download = await downloadPromise;
          const outputPath = join(outputDir, `${blobInfo.name}.wav`);
          await download.saveAs(outputPath);
          console.log(`Saved ${blobInfo.name} to ${outputPath}`);
        } catch (error) {
          console.error(`Error downloading ${blobInfo.name}:`, error);
        }
      }
    }

    // Assertions
    expect(result.modelLoaded).toBe(true);
    expect(result.separateError).toBeNull();
    expect(result.success).toBe(true);
    expect(result.outputBlobCount).toBe(2); // Should create 2 output files (vocals and instrumental)
  });
});
