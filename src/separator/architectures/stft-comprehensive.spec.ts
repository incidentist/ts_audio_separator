// Comprehensive STFT/ISTFT tests to verify the fix
import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { STFT } from './stft';

describe('STFT/ISTFT Comprehensive Tests', () => {
    beforeAll(async () => {
        await tf.setBackend('cpu');
        await tf.ready();
    });

    it('should reconstruct signal without conjugate issue (main bug fix)', async () => {
        // This test verifies that the conjugate bug has been fixed.
        // The bug was: applying tf.neg(imagT) before IRFFT, which caused
        // signal attenuation and phase distortion.
        const nFft = 512;
        const hopLength = 128;
        const dimF = 257; // No frequency truncation

        const stft = new STFT(nFft, hopLength, dimF);

        // Create a random signal
        const input = tf.randomNormal([1, 2, 1024]);

        // Round trip
        const spectrum = stft.forward(input);
        const reconstructed = stft.inverse(spectrum);

        // Compare
        const minLength = Math.min(input.shape[2], reconstructed.shape[2]);
        const inputTrimmed = input.slice([0, 0, 0], [1, 2, minLength]);
        const reconTrimmed = reconstructed.slice([0, 0, 0], [1, 2, minLength]);

        const mse = await tf.mean(tf.square(tf.sub(inputTrimmed, reconTrimmed))).data();
        const maxRatio = (await tf.max(tf.abs(reconTrimmed)).data())[0] /
                         (await tf.max(tf.abs(inputTrimmed)).data())[0];

        console.log('No freq truncation - MSE:', mse[0], 'Max ratio:', maxRatio);

        // Should be near-perfect reconstruction
        expect(mse[0]).toBeLessThan(1e-10);
        expect(maxRatio).toBeGreaterThan(0.99);
        expect(maxRatio).toBeLessThan(1.01);
    }, 30000);

    it('should handle frequency truncation gracefully', async () => {
        // Test with frequency truncation (as used in actual model)
        const nFft = 512;
        const hopLength = 128;
        const dimF = 200; // Less than nFft/2+1 = 257

        const stft = new STFT(nFft, hopLength, dimF);

        const input = tf.randomNormal([1, 2, 1024]);

        const spectrum = stft.forward(input);
        const reconstructed = stft.inverse(spectrum);

        const minLength = Math.min(input.shape[2], reconstructed.shape[2]);
        const inputTrimmed = input.slice([0, 0, 0], [1, 2, minLength]);
        const reconTrimmed = reconstructed.slice([0, 0, 0], [1, 2, minLength]);

        const mse = await tf.mean(tf.square(tf.sub(inputTrimmed, reconTrimmed))).data();
        const maxRatio = (await tf.max(tf.abs(reconTrimmed)).data())[0] /
                         (await tf.max(tf.abs(inputTrimmed)).data())[0];

        console.log('With freq truncation - MSE:', mse[0], 'Max ratio:', maxRatio);

        // Will have some error due to frequency truncation, but should still
        // preserve amplitude reasonably well.
        // With 78% of frequencies (200/257), some amplitude loss is expected.
        // The old conjugate bug caused ~34% attenuation. This should be better.
        expect(maxRatio).toBeGreaterThan(0.8);
        expect(maxRatio).toBeLessThan(1.2);
    }, 30000);

    it('should preserve phase information (no time reversal)', async () => {
        // The conjugate bug would cause phase distortion. This test verifies
        // that a linear ramp is preserved (not reversed).
        const nFft = 256;
        const hopLength = 64;
        const dimF = 129;

        const stft = new STFT(nFft, hopLength, dimF);

        // Create a ramp signal
        const rampData = Array.from({ length: 512 }, (_, i) => i / 512);
        const input = tf.tensor3d([[[...rampData]]]);

        const spectrum = stft.forward(input);
        const reconstructed = stft.inverse(spectrum);

        const reconData = await reconstructed.slice([0, 0, 0], [1, 1, 100]).array() as number[][][];
        const values = reconData[0][0];

        // Check that values are generally increasing (not reversed)
        let increasingCount = 0;
        for (let i = 1; i < values.length; i++) {
            if (values[i] > values[i - 1]) increasingCount++;
        }

        console.log('Phase preservation: increasing count =', increasingCount, 'out of', values.length - 1);
        expect(increasingCount).toBeGreaterThan(90); // Most should be increasing
    }, 30000);

    it('should maintain high SNR for sinusoidal signals', async () => {
        // Test with a known sinusoidal signal
        const sampleRate = 44100;
        const duration = 0.1;
        const numSamples = Math.floor(sampleRate * duration);
        const t = tf.linspace(0, duration, numSamples);

        const freq = 440;
        const signal = tf.sin(tf.mul(2 * Math.PI * freq, t));
        const stereoSignal = tf.stack([signal, signal]);
        const input = stereoSignal.expandDims(0);

        const nFft = 2048;
        const hopLength = 512;
        const dimF = 1025; // nFft/2 + 1

        const stft = new STFT(nFft, hopLength, dimF);

        const spectrum = stft.forward(input);
        const reconstructed = stft.inverse(spectrum);

        const minLength = Math.min(input.shape[2], reconstructed.shape[2]);
        const inputTrimmed = input.slice([0, 0, 0], [1, 2, minLength]);
        const reconTrimmed = reconstructed.slice([0, 0, 0], [1, 2, minLength]);

        const signalPower = await tf.mean(tf.square(inputTrimmed)).data();
        const mse = await tf.mean(tf.square(tf.sub(inputTrimmed, reconTrimmed))).data();
        const snrDb = 10 * Math.log10(signalPower[0] / mse[0]);

        console.log('Sinusoidal signal - SNR:', snrDb.toFixed(2), 'dB');
        expect(snrDb).toBeGreaterThan(60); // Should have excellent SNR
    }, 30000);

    it('should handle single channel input', async () => {
        const nFft = 512;
        const hopLength = 128;
        const dimF = 257;

        const stft = new STFT(nFft, hopLength, dimF);

        // Single channel input
        const input = tf.randomNormal([1, 1, 1024]);

        const spectrum = stft.forward(input);
        console.log('Single channel spectrum shape:', spectrum.shape);

        const reconstructed = stft.inverse(spectrum);
        console.log('Single channel reconstructed shape:', reconstructed.shape);

        const minLength = Math.min(input.shape[2], reconstructed.shape[2]);
        const inputTrimmed = input.slice([0, 0, 0], [1, 1, minLength]);
        const reconTrimmed = reconstructed.slice([0, 0, 0], [1, 1, minLength]);

        const mse = await tf.mean(tf.square(tf.sub(inputTrimmed, reconTrimmed))).data();
        console.log('Single channel MSE:', mse[0]);

        expect(mse[0]).toBeLessThan(1e-10);
    }, 30000);

    it('should handle stereo input with different channels', async () => {
        const nFft = 512;
        const hopLength = 128;
        const dimF = 257;

        const stft = new STFT(nFft, hopLength, dimF);

        // Stereo with different content in each channel
        const leftChannel = tf.randomNormal([1024]);
        const rightChannel = tf.randomNormal([1024]);
        const stereoSignal = tf.stack([leftChannel, rightChannel]);
        const input = stereoSignal.expandDims(0);

        const spectrum = stft.forward(input);
        const reconstructed = stft.inverse(spectrum);

        const minLength = Math.min(input.shape[2], reconstructed.shape[2]);
        const inputTrimmed = input.slice([0, 0, 0], [1, 2, minLength]);
        const reconTrimmed = reconstructed.slice([0, 0, 0], [1, 2, minLength]);

        const mse = await tf.mean(tf.square(tf.sub(inputTrimmed, reconTrimmed))).data();
        console.log('Stereo (different channels) MSE:', mse[0]);

        expect(mse[0]).toBeLessThan(1e-10);
    }, 30000);
});
