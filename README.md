# Noise Reduction using SpeechBrain

This project implements a noise reduction system using **SpeechBrain's MetricGAN+ model** to enhance audio recordings by reducing background noise.

## 🚀 Features
- **Removes background noise** from speech recordings.
- **Uses SpeechBrain's MetricGAN+ model** for audio enhancement.
- **Supports stereo-to-mono conversion** to ensure compatibility with the model.
- **Resamples audio to 16kHz** if necessary.
- **Saves the enhanced audio output** after processing.

## 📌 Requirements
Make sure you have the following dependencies installed:
```bash
pip install torch torchaudio speechbrain librosa matplotlib
```

## 📂 Project Structure
```
├── recorded_audio.wav         # Sample noisy audio file
├── converted_audio2.wav       # Denoised output audio file
├── noise_reduction.py         # Main script for noise reduction
├── README.md                  # Project documentation
```

## 🛠️ Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Run the Noise Reduction Script
```bash
python noise_reduction.py
```
This will process `recorded_audio.wav` and generate a cleaned-up version `converted_audio2.wav`.

### 3️⃣ Listen to the Output
Use any media player or run:
```bash
python -m playsound converted_audio2.wav
```

## 📜 Code Overview
```python
import torchaudio
import torch
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Load Model
model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus"
)

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("recorded_audio.wav")
waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if needed
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

# Run Noise Reduction
waveform = waveform.unsqueeze(0)
lengths = torch.tensor([waveform.shape[-1] / waveform.shape[-1]], dtype=torch.float32)
enhanced_waveform = model.enhance_batch(waveform, lengths=lengths).squeeze(0)

# Normalize and Save Output
enhanced_waveform /= enhanced_waveform.abs().max()
torchaudio.save("converted_audio2.wav", enhanced_waveform, 16000)
```

## 🔥 Improvements & Future Work
- Add support for **real-time noise reduction**.
- Experiment with **different denoising models** for comparison.
- Implement **GUI for easier usability**.


