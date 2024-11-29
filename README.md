# CosmicQuakes 🌌  
**Seismic Detection Across the Solar System**  

CosmicQuakes is a project that utilizes advanced signal processing and machine learning techniques to detect and analyze seismic activities across planetary bodies in the Solar System.

## Features
- Signal preprocessing using Empirical Mode Decomposition (EMD) and Bandpass Filtering.
- Frequency-domain analysis with Fourier Transform (FFT).
- Event detection using CNN-LSTM and optional U-Net architectures.
- Post-processing verification with auto-encoders to enhance accuracy.

## Repository Structure
- **data/**: Contains raw and processed seismic datasets.
- **models/**: Stores trained models.
- **notebooks/**: Jupyter notebooks for analysis and experimentation.
- **src/**: Source code for signal processing, modeling, and utilities.
- **tests/**: Scripts for testing various components.

## Requirements
Install the necessary libraries with:
```bash
pip install -r requirements.txt
