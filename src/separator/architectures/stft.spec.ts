// src/stft.spec.ts
import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { STFT } from './stft';

describe('STFT', () => {
    beforeAll(async () => {

        // Set TensorFlow.js backend to WebGL for better performance
        await tf.setBackend('cpu');
        await tf.ready(); // Wait for backend to be ready

        console.log('TensorFlow backend ready:', tf.getBackend());

    });

    it('should reconstruct signal through forward and inverse transform', async () => {
        console.log("starting test");
        // Create a test signal - mix of sinusoids
        const sampleRate = 44100;
        const duration = 0.1; // 1 second
        const numSamples = Math.floor(sampleRate * duration);
        const t = tf.linspace(0, duration, numSamples);

        // Create a test signal with multiple frequency components
        const freq1 = 440; // A4
        const freq2 = 880; // A5
        const signal1 = tf.sin(tf.mul(2 * Math.PI * freq1, t));
        const signal2 = tf.mul(0.5, tf.sin(tf.mul(2 * Math.PI * freq2, t)));
        const monoSignal = tf.add(signal1, signal2);

        // Create stereo signal [2, numSamples]
        const stereoSignal = tf.stack([monoSignal, monoSignal]);

        // Add batch dimension [1, 2, numSamples]
        const input = stereoSignal.expandDims(0);

        // Initialize STFT with parameters from your model
        const nFft = 5120;
        const hopLength = 1024;
        const dimF = 2048;
        const stft = new STFT(nFft, hopLength, dimF);

        // Forward pass
        console.log('Running forward');
        const spectrum = stft.forward(input);
        console.log('Spectrum shape:', spectrum.shape);

        // Calculate expected number of frames
        const paddedLength = numSamples + 2 * Math.floor(nFft / 2); // center padding
        const expectedFrames = Math.floor((paddedLength - nFft) / hopLength) + 1;
        console.log('Expected frames:', expectedFrames);
        console.log('Actual frames:', spectrum.shape[3]);

        // Inverse pass
        const reconstructed = stft.inverse(spectrum, input.shape[input.shape.length - 1]);
        console.log('Reconstructed shape:', reconstructed.shape);
        console.log('Expected shape:', input.shape);


        // Check if shapes match
        expect(reconstructed.shape).toEqual(input.shape);

        // Calculate reconstruction error
        const diff = tf.sub(reconstructed, input);
        const mse = tf.mean(tf.square(diff));
        const mseValue = await mse.data();
        console.log('MSE:', mseValue[0]);

        // Calculate SNR (Signal-to-Noise Ratio)
        const signalPower = tf.mean(tf.square(input));
        const noisePower = mse;
        const snr = tf.div(signalPower, noisePower);
        const snrDb = tf.mul(10, tf.log(snr).div(tf.log(10)));
        const snrValue = await snrDb.data();
        console.log('SNR (dB):', snrValue[0]);

        // The reconstruction should be nearly perfect
        expect(mseValue[0]).toBeLessThan(1e-6);
        expect(snrValue[0]).toBeGreaterThan(60); // Should have > 60dB SNR

        // Also check that the audio values are in reasonable range
        const maxReconstructed = tf.max(tf.abs(reconstructed));
        const maxReconstructedValue = await maxReconstructed.data();
        const maxInput = tf.max(tf.abs(input));
        const maxInputValue = await maxInput.data();

        console.log('Max input value:', maxInputValue[0]);
        console.log('Max reconstructed value:', maxReconstructedValue[0]);

        // Values should be similar
        expect(Math.abs(maxReconstructedValue[0] - maxInputValue[0])).toBeLessThan(0.1);

        // Clean up
        t.dispose();
        signal1.dispose();
        signal2.dispose();
        monoSignal.dispose();
        stereoSignal.dispose();
        input.dispose();
        spectrum.dispose();
        reconstructed.dispose();
        diff.dispose();
        mse.dispose();
        signalPower.dispose();
        noisePower.dispose();
        snr.dispose();
        snrDb.dispose();
        maxReconstructed.dispose();
        maxInput.dispose();
    });

    // it('should handle model chunk size correctly', async () => {
    //     console.log("starting test 2");

    //     // Use the exact chunk size from your model
    //     const chunkSize = 261120;
    //     const nFft = 5120;
    //     const hopLength = 1024;
    //     const dimF = 2048;

    //     // Create random audio-like data
    //     const audioData = tf.randomNormal([1, 2, chunkSize], 0, 0.1);

    //     const stft = new STFT(nFft, hopLength, dimF);

    //     // Forward and inverse
    //     const spectrum = stft.forward(audioData);
    //     const reconstructed = stft.inverse(spectrum);

    //     // Trim to match input size (if needed)
    //     const trimmedReconstructed = reconstructed.slice([0, 0, 0], audioData.shape);

    //     // Calculate error
    //     const diff = tf.sub(trimmedReconstructed, audioData);
    //     const mse = tf.mean(tf.square(diff));
    //     const mseValue = await mse.data();

    //     console.log('Chunk size test MSE:', mseValue[0]);
    //     expect(mseValue[0]).toBeLessThan(1e-5);

    //     // Clean up
    //     audioData.dispose();
    //     spectrum.dispose();
    //     reconstructed.dispose();
    //     trimmedReconstructed.dispose();
    //     diff.dispose();
    //     mse.dispose();
    // });

    // it('should preserve phase information correctly', async () => {
    //     console.log("starting test 3");

    //     // Create a chirp signal (frequency sweep)
    //     const sampleRate = 44100;
    //     const duration = 0.5;
    //     const numSamples = Math.floor(sampleRate * duration);
    //     const t = tf.linspace(0, duration, numSamples);

    //     // Chirp from 100Hz to 1000Hz
    //     const phase = tf.mul(2 * Math.PI, tf.mul(t, tf.add(100, tf.mul(900, tf.div(t, duration)))));
    //     const chirp = tf.sin(phase);

    //     // Create stereo signal
    //     const stereoSignal = tf.stack([chirp, chirp]);
    //     const input = stereoSignal.expandDims(0);

    //     // Initialize STFT
    //     const nFft = 5120;
    //     const hopLength = 1024;
    //     const dimF = 2048;
    //     const stft = new STFT(nFft, hopLength, dimF);

    //     // Forward and inverse
    //     const spectrum = stft.forward(input);
    //     const reconstructed = stft.inverse(spectrum);

    //     // Calculate correlation between input and output
    //     const inputFlat = input.flatten();
    //     const reconstructedFlat = reconstructed.flatten();

    //     // Normalize signals
    //     const inputNorm = tf.div(inputFlat, tf.norm(inputFlat));
    //     const reconstructedNorm = tf.div(reconstructedFlat, tf.norm(reconstructedFlat));

    //     // Calculate correlation
    //     const correlation = tf.sum(tf.mul(inputNorm, reconstructedNorm));
    //     const correlationValue = await correlation.data();

    //     console.log('Phase preservation correlation:', correlationValue[0]);
    //     expect(correlationValue[0]).toBeGreaterThan(0.99); // Should be very close to 1

    //     // Clean up
    //     t.dispose();
    //     phase.dispose();
    //     chirp.dispose();
    //     stereoSignal.dispose();
    //     input.dispose();
    //     spectrum.dispose();
    //     reconstructed.dispose();
    //     inputFlat.dispose();
    //     reconstructedFlat.dispose();
    //     inputNorm.dispose();
    //     reconstructedNorm.dispose();
    //     correlation.dispose();
    // });

    // it('should handle edge cases correctly', async () => {
    //     const nFft = 5120;
    //     const hopLength = 1024;
    //     const dimF = 2048;
    //     const stft = new STFT(nFft, hopLength, dimF);

    //     // Test with very short signal
    //     const shortSignal = tf.randomNormal([1, 2, 8192], 0, 0.1);
    //     const shortSpectrum = stft.forward(shortSignal);
    //     const shortReconstructed = stft.inverse(shortSpectrum);

    //     // Should handle without errors
    //     expect(shortReconstructed.shape[0]).toBe(1);
    //     expect(shortReconstructed.shape[1]).toBe(2);

    //     // Test with silence
    //     const silence = tf.zeros([1, 2, 44100]);
    //     const silenceSpectrum = stft.forward(silence);
    //     const silenceReconstructed = stft.inverse(silenceSpectrum);

    //     const silenceMax = tf.max(tf.abs(silenceReconstructed));
    //     const silenceMaxValue = await silenceMax.data();

    //     console.log('Silence max value:', silenceMaxValue[0]);
    //     expect(silenceMaxValue[0]).toBeLessThan(1e-10); // Should be essentially zero

    //     // Clean up
    //     shortSignal.dispose();
    //     shortSpectrum.dispose();
    //     shortReconstructed.dispose();
    //     silence.dispose();
    //     silenceSpectrum.dispose();
    //     silenceReconstructed.dispose();
    //     silenceMax.dispose();
    // });

    // it('should match expected spectrum characteristics', async () => {
    //     // Create a pure tone to verify frequency bin mapping
    //     const sampleRate = 44100;
    //     const duration = 1.0;
    //     const numSamples = Math.floor(sampleRate * duration);
    //     const t = tf.linspace(0, duration, numSamples);

    //     // 1000 Hz pure tone
    //     const freq = 1000;
    //     const pureTone = tf.sin(tf.mul(2 * Math.PI * freq, t));
    //     const stereoSignal = tf.stack([pureTone, pureTone]);
    //     const input = stereoSignal.expandDims(0);

    //     const nFft = 5120;
    //     const hopLength = 1024;
    //     const dimF = 2048;
    //     const stft = new STFT(nFft, hopLength, dimF);

    //     const spectrum = stft.forward(input);

    //     // Extract magnitude spectrum for first channel
    //     const realPart = spectrum.slice([0, 0, 0, 0], [1, 1, -1, -1]);
    //     const imagPart = spectrum.slice([0, 1, 0, 0], [1, 1, -1, -1]);

    //     const magnitude = tf.sqrt(tf.add(tf.square(realPart), tf.square(imagPart)));

    //     // Find the peak frequency bin
    //     const meanOverTime = tf.mean(magnitude, 3);
    //     const peakBin = tf.argMax(meanOverTime.squeeze(), 0);
    //     const peakBinValue = await peakBin.data();

    //     // Calculate expected bin
    //     const binFrequency = sampleRate / nFft;
    //     const expectedBin = Math.round(freq / binFrequency);

    //     console.log('Peak bin:', peakBinValue[0]);
    //     console.log('Expected bin:', expectedBin);

    //     // Should be close to expected bin
    //     expect(Math.abs(peakBinValue[0] - expectedBin)).toBeLessThan(3);

    //     // Clean up
    //     t.dispose();
    //     pureTone.dispose();
    //     stereoSignal.dispose();
    //     input.dispose();
    //     spectrum.dispose();
    //     realPart.dispose();
    //     imagPart.dispose();
    //     magnitude.dispose();
    //     meanOverTime.dispose();
    //     peakBin.dispose();
    // });
});