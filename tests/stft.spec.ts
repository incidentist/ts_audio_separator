import { test, expect } from '@playwright/test';
import * as tf from '@tensorflow/tfjs';
import { STFT } from './stft';

test('STFT forward and inverse should reconstruct signal', async () => {
    // Create a test signal - mix of sinusoids
    const sampleRate = 44100;
    const duration = 1.0; // 1 second
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
    const spectrum = stft.forward(input);
    console.log('Spectrum shape:', spectrum.shape);

    // Inverse pass
    const reconstructed = stft.inverse(spectrum);
    console.log('Reconstructed shape:', reconstructed.shape);

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

// Also test with actual audio-like chunk size
test('STFT round-trip with model chunk size', async () => {
    // Use the exact chunk size from your model
    const chunkSize = 261120;
    const nFft = 5120;
    const hopLength = 1024;
    const dimF = 2048;

    // Create random audio-like data
    const audioData = tf.randomNormal([1, 2, chunkSize], 0, 0.1);

    const stft = new STFT(nFft, hopLength, dimF);

    // Forward and inverse
    const spectrum = stft.forward(audioData);
    const reconstructed = stft.inverse(spectrum);

    // Trim to match input size (if needed)
    const trimmedReconstructed = reconstructed.slice([0, 0, 0], audioData.shape);

    // Calculate error
    const diff = tf.sub(trimmedReconstructed, audioData);
    const mse = tf.mean(tf.square(diff));
    const mseValue = await mse.data();

    console.log('Chunk size test MSE:', mseValue[0]);
    expect(mseValue[0]).toBeLessThan(1e-5);

    // Clean up
    audioData.dispose();
    spectrum.dispose();
    reconstructed.dispose();
    trimmedReconstructed.dispose();
    diff.dispose();
    mse.dispose();
});