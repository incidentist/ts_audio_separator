import ndarray from 'ndarray';
import fft from 'ndarray-fft';

// Hann window function
function hannWindow(size) {
    const win = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
    }
    return win;
}

// Center-pad input signal
function centerPad(signal, pad) {
    const padded = new Float32Array(signal.length + 2 * pad);
    for (let i = 0; i < signal.length; i++) {
        padded[pad + i] = signal[i];
    }
    return padded;
}

// Remove center padding
function removeCenterPad(signal, pad) {
    return signal.subarray(pad, signal.length - pad);
}

// Main STFT function that returns both magnitude and phase
function stft(input, frameLength, frameStep, fftLength = null, windowFn = hannWindow, center = true) {
    let signal = input instanceof Float32Array ? input : new Float32Array(input);
    fftLength = fftLength || frameLength;

    if (center) {
        const pad = Math.floor(frameLength / 2);
        signal = centerPad(signal, pad);
    }

    const window = windowFn(frameLength);
    const numFrames = Math.floor((signal.length - frameLength) / frameStep) + 1;

    const magnitude = [];
    const phase = [];

    for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * frameStep;
        const frameData = new Float32Array(fftLength);

        for (let i = 0; i < frameLength; i++) {
            frameData[i] = signal[start + i] * window[i];
        }

        const re = ndarray(new Float32Array(fftLength), [fftLength]);
        const im = ndarray(new Float32Array(fftLength), [fftLength]);

        for (let i = 0; i < fftLength; i++) {
            re.set(i, frameData[i]);
            im.set(i, 0);
        }

        fft(1, re, im); // Forward FFT

        // Use only half of the spectrum (real signals are symmetric in frequency domain)
        const frameMagnitude = new Float32Array(Math.floor(fftLength / 2) + 1);
        const framePhase = new Float32Array(Math.floor(fftLength / 2) + 1);

        for (let i = 0; i <= Math.floor(fftLength / 2); i++) {
            const real = re.get(i);
            const imag = im.get(i);
            frameMagnitude[i] = Math.sqrt(real * real + imag * imag);
            framePhase[i] = Math.atan2(imag, real);
        }

        magnitude.push(frameMagnitude);
        phase.push(framePhase);
    }

    return { magnitude, phase }; // Return object with magnitude and phase arrays
}

// Inverse STFT compatible with the format returned by our stft function
function istft(magnitude, phase, frameLength, frameStep, fftLength = null, windowFn = hannWindow, center = true) {
    fftLength = fftLength || frameLength;
    const window = windowFn(frameLength);
    const signalLength = frameStep * (magnitude.length - 1) + frameLength;

    const signal = new Float32Array(signalLength);
    const windowSum = new Float32Array(signalLength);

    for (let frame = 0; frame < magnitude.length; frame++) {
        const start = frame * frameStep;
        const re = ndarray(new Float32Array(fftLength), [fftLength]);
        const im = ndarray(new Float32Array(fftLength), [fftLength]);

        // Convert magnitude and phase to complex representation
        for (let i = 0; i <= Math.floor(fftLength / 2); i++) {
            const mag = magnitude[frame][i];
            const ph = phase[frame][i];
            re.set(i, mag * Math.cos(ph));
            im.set(i, mag * Math.sin(ph));

            // Mirror for the second half (except DC and Nyquist)
            if (i > 0 && i < Math.floor(fftLength / 2)) {
                re.set(fftLength - i, mag * Math.cos(ph));
                im.set(fftLength - i, -mag * Math.sin(ph)); // Conjugate for the mirror
            }
        }

        fft(-1, re, im); // Inverse FFT

        for (let i = 0; i < frameLength; i++) {
            const value = re.get(i) * window[i];
            signal[start + i] += value;
            windowSum[start + i] += window[i] * window[i];
        }
    }

    // Normalize
    for (let i = 0; i < signal.length; i++) {
        if (windowSum[i] > 1e-8) {
            signal[i] /= windowSum[i];
        }
    }

    if (center) {
        const pad = Math.floor(frameLength / 2);
        return removeCenterPad(signal, pad);
    } else {
        return signal;
    }
}

export { stft, istft };
