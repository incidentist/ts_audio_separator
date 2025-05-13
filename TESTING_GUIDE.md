# Running the MDX Separator Test

## Summary

The MDX separator is now complete and can save separated audio files as playable WAV files. However, due to browser security restrictions with blob URLs and large file conversions, I've created several ways to test and save the output:

## Test Options

### 1. Manual Browser Test (Recommended)

Open `test-downloads.html` in your browser:

1. Start the dev server: `npm run dev`
2. Open http://localhost:5173/test-downloads.html
3. Click "Run Separation Test"
4. Once complete, you can:
   - Play the separated tracks directly in the browser
   - Click individual download buttons to save vocals or instrumental
   - Click "Download All Files" to save both tracks

### 2. Automated Test with Direct Download

The test file has been updated to use Playwright's download handling:

```bash
npm test mdx-separator
```

This test will:
- Process the audio file
- Display blob information in the console
- Attempt to trigger downloads programmatically

### 3. Simple Test Page

You can also use `separation-test.html` for a simpler interface:

1. Open http://localhost:5173/separation-test.html
2. Click "Start Separation Test"
3. Download the separated files using the download buttons

## What's Working

- ✓ Audio separation using the MDX model
- ✓ Blob creation with proper WAV format
- ✓ Playback in browser audio elements
- ✓ Manual download via button clicks
- ✓ File metadata (size, name) tracking

## Known Issues

- Large file base64 conversion causes stack overflow (fixed by removing it)
- Direct programmatic downloads may be blocked by browser security
- Test automation of downloads requires user interaction

## Output Format

The separated files are saved as:
- `vocals.wav` - Isolated vocal track
- `instrumental.wav` - Instrumental track without vocals

Both are standard 16-bit WAV files at 44100 Hz sample rate that can be played in any audio player.

## Next Steps

To fully automate the test downloads, you might want to:
1. Use a headless browser with relaxed security settings
2. Configure Playwright to handle downloads automatically
3. Use a server-side component to save files directly

For now, the manual browser test provides the most reliable way to save the separated audio files.
