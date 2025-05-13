// This file is loaded by the test to make our library available in the browser
import * as WebDemix2 from './index';

// Make it available on the window object for the test
(window as any).WebDemix2 = WebDemix2;

// Also expose the ndarray STFT for testing
import('./utils/stft-ndarray.js').then(module => {
  (window as any).ndarraySTFT = module;
});
