import { test, expect } from '@playwright/test';

test.describe('STFT Performance Test', () => {
  test.skip('benchmark custom STFT implementation', async ({ page }) => {
    page.on('console', msg => console.log('Browser console:', msg.text()));
    page.on('pageerror', error => console.log('Page error:', error));

    await page.goto('http://localhost:5173/test-page.html');

    // Wait for page to load
    await page.waitForTimeout(1000);

    // Execute performance test in browser context
    const results = await page.evaluate(async () => {
      try {
        // Wait for library to load
        const checkLibrary = () => new Promise<void>((resolve) => {
          const interval = setInterval(() => {
            if ((window as any).WebDemix2) {
              clearInterval(interval);
              resolve();
            }
          }, 100);
        });

        await checkLibrary();
        console.log('WebDemix2 loaded');

        // Import our custom STFT
        const { STFT } = (window as any).WebDemix2;

        // Test parameters
        const configurations = [
          { nFft: 2048, hopLength: 512, dimF: 1024, name: 'Small' },
          { nFft: 5120, hopLength: 1024, dimF: 2048, name: 'Medium (MDX)' },
          { nFft: 7680, hopLength: 1024, dimF: 3072, name: 'Large (MDX)' },
        ];

        const results: any = {};

        for (const config of configurations) {
          console.log(`Testing ${config.name} configuration...`);

          // Create test signal (2 channels, varying durations)
          const sampleRate = 44100;
          const durations = [0.1, 0.5, 1.0]; // Test different durations

          const configResults: any = {
            nFft: config.nFft,
            hopLength: config.hopLength,
            dimF: config.dimF,
            durations: {}
          };

          for (const duration of durations) {
            const samples = Math.floor(sampleRate * duration);
            const testSignal = [
              new Float32Array(samples),
              new Float32Array(samples)
            ];

            // Generate test signal
            for (let i = 0; i < samples; i++) {
              testSignal[0][i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
              testSignal[1][i] = Math.sin(2 * Math.PI * 554 * i / sampleRate) * 0.3;
            }

            // Initialize STFT
            const stft = new STFT(config.nFft, config.hopLength, config.dimF);

            // Warm up run
            stft.forward(testSignal);

            // Benchmark forward transform
            const iterations = 10;
            let totalForwardTime = 0;
            let totalInverseTime = 0;

            for (let i = 0; i < iterations; i++) {
              const forwardStart = performance.now();
              const spec = stft.forward(testSignal);
              totalForwardTime += performance.now() - forwardStart;

              const inverseStart = performance.now();
              const reconstructed = stft.inverse(spec);
              totalInverseTime += performance.now() - inverseStart;
            }

            // Calculate metrics
            const avgForwardTime = totalForwardTime / iterations;
            const avgInverseTime = totalInverseTime / iterations;
            const numFrames = Math.floor((samples - config.nFft) / config.hopLength) + 1;

            // Calculate throughput (samples per second)
            const forwardThroughput = (samples * 2) / (avgForwardTime / 1000); // stereo samples/sec
            const inverseThroughput = (samples * 2) / (avgInverseTime / 1000);

            configResults.durations[duration] = {
              samples,
              numFrames,
              avgForwardTime,
              avgInverseTime,
              forwardThroughput,
              inverseThroughput,
              msPerFrame: avgForwardTime / numFrames
            };
          }

          results[config.name] = configResults;
        }

        // Test MDX-specific chunk processing
        const mdxConfig = { nFft: 7680, hopLength: 1024, dimF: 3072 };
        const segmentSize = 256;
        const chunkSize = (segmentSize - 1) * mdxConfig.hopLength + mdxConfig.nFft;

        const mdxChunk = [
          new Float32Array(chunkSize),
          new Float32Array(chunkSize)
        ];

        // Fill with random data
        for (let i = 0; i < chunkSize; i++) {
          mdxChunk[0][i] = Math.random() * 0.1 - 0.05;
          mdxChunk[1][i] = Math.random() * 0.1 - 0.05;
        }

        const mdxStft = new STFT(mdxConfig.nFft, mdxConfig.hopLength, mdxConfig.dimF);
        const mdxStart = performance.now();
        const mdxSpec = mdxStft.forward(mdxChunk, segmentSize);
        const mdxTime = performance.now() - mdxStart;

        results.mdxChunkTest = {
          chunkSize,
          targetFrames: segmentSize,
          actualFrames: mdxSpec[0][0].length,
          processingTime: mdxTime,
          framesMatch: mdxSpec[0][0].length === segmentSize
        };

        return results;
      } catch (error: any) {
        console.error('Error in performance test:', error);
        throw error;
      }
    });

    console.log('\n=== Custom STFT Performance Results ===\n');

    // Display results for each configuration
    for (const [configName, configResults] of Object.entries(results)) {
      if (configName === 'mdxChunkTest') continue;

      const config = configResults as any;
      console.log(`\n${configName} Configuration (nFft=${config.nFft}, hop=${config.hopLength}):`);

      for (const [duration, metrics] of Object.entries(config.durations)) {
        const m = metrics as any;
        console.log(`\n  ${duration}s audio (${m.samples} samples, ${m.numFrames} frames):`);
        console.log(`    Forward:  ${m.avgForwardTime.toFixed(2)}ms (${(m.forwardThroughput / 1000000).toFixed(2)} MSamples/sec)`);
        console.log(`    Inverse:  ${m.avgInverseTime.toFixed(2)}ms (${(m.inverseThroughput / 1000000).toFixed(2)} MSamples/sec)`);
        console.log(`    Per frame: ${m.msPerFrame.toFixed(3)}ms`);
      }
    }

    // Display MDX chunk test results
    const mdxTest = results.mdxChunkTest;
    console.log('\n\nMDX Chunk Processing Test:');
    console.log(`  Chunk size: ${mdxTest.chunkSize} samples`);
    console.log(`  Target frames: ${mdxTest.targetFrames}`);
    console.log(`  Actual frames: ${mdxTest.actualFrames}`);
    console.log(`  Processing time: ${mdxTest.processingTime.toFixed(2)}ms`);
    console.log(`  Frames match: ${mdxTest.framesMatch ? 'YES ✓' : 'NO ✗'}`);

    // Performance summary
    const mediumConfig = results['Medium (MDX)'];
    const oneSecMetrics = mediumConfig.durations['1'];
    console.log('\n\nPerformance Summary (1s stereo audio, MDX config):');
    console.log(`  Forward transform: ${oneSecMetrics.avgForwardTime.toFixed(2)}ms`);
    console.log(`  Inverse transform: ${oneSecMetrics.avgInverseTime.toFixed(2)}ms`);
    console.log(`  Total round-trip: ${(oneSecMetrics.avgForwardTime + oneSecMetrics.avgInverseTime).toFixed(2)}ms`);
    console.log(`  Real-time factor: ${(1000 / (oneSecMetrics.avgForwardTime + oneSecMetrics.avgInverseTime)).toFixed(2)}x`);

    // Assertions
    expect(mdxTest.framesMatch).toBe(true);
    expect(mdxTest.actualFrames).toBe(256);

    // Performance assertions (reasonable expectations)
    expect(oneSecMetrics.avgForwardTime).toBeLessThan(5000); // Should process in under 5 seconds
    expect(oneSecMetrics.avgInverseTime).toBeLessThan(3000); // Inverse usually faster
  });

  test('verify STFT accuracy', async ({ page }) => {
    page.on('console', msg => console.log('Browser console:', msg.text()));

    await page.goto('http://localhost:5173/test-page.html');
    await page.waitForTimeout(1000);

    const results = await page.evaluate(async () => {
      const { STFT } = (window as any).WebDemix2;

      // Test with a known signal
      const sampleRate = 44100;
      const duration = 0.1; // 100ms
      const samples = Math.floor(sampleRate * duration);
      const frequency = 1000; // 1kHz

      const signal = [
        new Float32Array(samples),
        new Float32Array(samples)
      ];

      // Generate pure sine wave
      for (let i = 0; i < samples; i++) {
        const value = Math.sin(2 * Math.PI * frequency * i / sampleRate);
        signal[0][i] = value;
        signal[1][i] = value;
      }

      // Process through STFT
      const stft = new STFT(2048, 512, 1024);
      const spec = stft.forward(signal);
      const reconstructed = stft.inverse(spec);

      // Calculate reconstruction error
      let error = 0;
      let signalPower = 0;

      for (let ch = 0; ch < 2; ch++) {
        const minLen = Math.min(signal[ch].length, reconstructed[ch].length);
        for (let i = 0; i < minLen; i++) {
          error += Math.pow(signal[ch][i] - reconstructed[ch][i], 2);
          signalPower += Math.pow(signal[ch][i], 2);
        }
      }

      const rmse = Math.sqrt(error / (samples * 2));
      const snr = 10 * Math.log10(signalPower / error);

      // Find peak frequency bin
      let maxMag = 0;
      let peakBin = 0;
      const midFrame = Math.floor(spec[0][0].length / 2);

      for (let bin = 0; bin < spec[0].length; bin++) {
        const mag = Math.sqrt(
          spec[0][bin][midFrame][0] ** 2 +
          spec[0][bin][midFrame][1] ** 2
        );
        if (mag > maxMag) {
          maxMag = mag;
          peakBin = bin;
        }
      }

      const binFreq = (peakBin * sampleRate) / 2048;

      return {
        rmse,
        snr,
        peakBin,
        binFreq,
        expectedFreq: frequency,
        freqError: Math.abs(binFreq - frequency)
      };
    });

    console.log('\n=== STFT Accuracy Test ===\n');
    console.log(`RMSE: ${results.rmse.toExponential(3)}`);
    console.log(`SNR: ${results.snr.toFixed(2)} dB`);
    console.log(`Peak frequency: ${results.binFreq.toFixed(2)} Hz (expected: ${results.expectedFreq} Hz)`);
    console.log(`Frequency error: ${results.freqError.toFixed(2)} Hz`);

    // Assertions
    expect(results.rmse).toBeLessThan(0.001); // Very low reconstruction error
    expect(results.snr).toBeGreaterThan(40); // Good signal-to-noise ratio
    expect(results.freqError).toBeLessThan(50); // Frequency resolution limited by FFT size
  });
});
