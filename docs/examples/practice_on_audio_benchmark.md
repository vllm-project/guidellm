# GuideLLM Audio Benchmark Testing Practice

Do audio benchmark test based on GuideLLM, covering the complete workflow from environment setup to test execution and analysis.

## Getting Started

### 📦 1. Benchmark Testing Environment Setup

#### 1.1 Create Conda Environment (recommended)

```bash
conda create -n guidellm-bench python=3.11 -y
conda activate guidellm-bench
```

#### 1.2 Install GuideLLM

```bash
git clone https://github.com/vllm-project/guidellm.git
cd guidellm
pip install guidellm
```

For more detailed instructions, refer to [GuideLLM README](https://github.com/vllm-project/guidellm/blob/main/README.md).

#### 1.3 Verify Installation

```bash
guidellm --help
```

#### 1.4 Start Audio-Compatible API Server
In a new terminal, start the mock server using:

```bash
guidellm mock-server --host 0.0.0.0 --port 8080
```

#### 1.5 Generate local mp3 file for validation
1. Install Text-to-Speech Tool
```bash
$ pip install gTTS
```
2. Generate Test Audio File
```bash
gtts-cli 'hello' --output hello.mp3
```
Reference: Detailed gTTS documentation is available at [gTTS](https://pypi.org/project/gTTS/) PyPI page.


#### 1.6 API Function Verification

```bash
# Basic transcription request
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions
-F "file=@/hello.mp3"
-F "model=whisper-1"
-F "language=zh"

# Advanced transcription parameters
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions
-F "file=@/hello.mp3"
-F "model=whisper-large-v3"
-F "language=en"
-F "prompt=This is a technical demonstration"
-F "temperature=0.3"
-F "response_format=verbose_json"

# Text format response
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions
-F "file=@/hello.mp3"
-F "response_format=text"

# Audio translation
curl -X POST http://127.0.0.1:8080/v1/audio/translations
-F "file=@/hello.mp3"
```

#### 1.7 Benchmark Test Data Generation

Use the following Python script to generate test audio files and corresponding metadata in batch:

```python

from itertools import count
import numpy as np
import pandas as pd
import wave
import struct
import click
import os

@click.command()
@click.option(
    "--count",
    default=0,
    type=int,
    help="Number of audio files to generate.",
)
def generate_and_save_wav_with_metadata(count: int):
    # create subdirectory to save wav files and metadata in the current directory
    current_path = os.path.abspath(os.getcwd())
    subdir_name = current_path + "/audio_test_output"
    if not os.path.exists(subdir_name):
        os.mkdir(subdir_name)

    """Generate WAV files and save metadata to CSV"""
    # audio parameters
    sample_rate = 44100
    duration = 3.0
    frequency = 523.25 # C5 note frequency

    metadata_list = []
    for i in range(count):
        # generate a sine wave at the specified frequency
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        # transfer to int16 PCM format
        audio_int16 = np.int16(audio_data * 32767)

        audio_filename = subdir_name + f'/test_audio_{i}.wav'

        # save to WAV file
        with wave.open(audio_filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # single channel
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"WAV files saved to: {audio_filename}")

        audio_filename = subdir_name + f'/test_audio_{i}.wav'
        metadata_list.append({
            'audio': audio_filename,
            'sample_rate': sample_rate,
            'duration': duration,
            'frequency_hz': frequency,
            'channels': 1,
            'bits_per_sample': 16,
            'num_samples': int(sample_rate * duration),
            'max_amplitude': 0.5,
            'rms': float(np.sqrt(np.mean((0.5 * np.sin(2 * np.pi * frequency * np.linspace(0, duration, int(sample_rate * duration), endpoint=False)))**2)))
        })

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(current_path + '/audio_metadata.csv', mode='a', index=False)

    print(f"audio metadata saved to: {current_path}/audio_metadata.csv")


#CMD example: python3 audio_test4.py --count 5
if __name__ == "__main__":
    generate_and_save_wav_with_metadata()

```

Run the script to generate 5 test audio files:

```
% python3 audio_test4.py --count 5
WAV files saved to: ${abs_path}/audio_test_output/test_audio_0.wav
WAV files saved to: ${abs_path}/audio_test_output/test_audio_1.wav
WAV files saved to: ${abs_path}/audio_test_output/test_audio_2.wav
WAV files saved to: ${abs_path}/audio_test_output/test_audio_3.wav
WAV files saved to: ${abs_path}/audio_test_output/test_audio_4.wav
audio metadata saved to: ${abs_path}/audio_metadata.csv
```

______________________________________________________________________

## 🚀 2. Running Benchmarks

```bash
guidellm benchmark \
    --target "http://localhost:8080" \
    --request-type "audio_transcriptions" \
    --rate-type "throughput" \
    --rate 1 \
    --max-requests 5 \
    --data "./audio_metadata.csv"
```

![Benchmark result](../assets/audio-benchmark-result.gif)

______________________________________________________________________

The audio benchmark test complete.
