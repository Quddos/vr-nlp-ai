# VR-NLP-AI

An Immersive Web SDK (IWSDK) application that integrates neural machine translation with attention mechanism for VR/AR experiences.

## Features

- **Neural Machine Translation**: Implements the Bahdanau attention mechanism from the paper "Neural Machine Translation by Jointly Learning to Align and Translate"
- **VR/AR Interface**: Interactive UI in immersive environments
- **Real-time Translation**: API-based translation service

## Architecture

- **Frontend**: IWSDK (TypeScript/JavaScript) for VR/AR interface
- **Backend**: Python Flask API serving PyTorch neural translation model
- **Model**: Sequence-to-sequence with attention (Encoder-Decoder architecture)

## Setup and Running

### Prerequisites

- Node.js (>=20.19.0)
- Python 3.8+
- npm

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install torch numpy flask
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

### Running the Application

1. **Start the Python translation server:**
   ```bash
   python app.py
   ```
   This will train the model (if not already trained) and start the Flask API on port 5000.

2. **Start the IWSDK development server:**
   ```bash
   npm run dev
   ```
   This starts the VR application on https://localhost:8081/

3. **Access the application:**
   - Open https://localhost:8081/ in a WebXR-compatible browser
   - Enter VR mode using the "Enter XR" button
   - Use the translation panel to input English text and get translations

## Model Details

The implementation includes:

- **Encoder**: Bidirectional GRU
- **Decoder**: GRU with Bahdanau attention
- **Attention Mechanism**: Content-based attention for alignment
- **Training**: Teacher forcing with cross-entropy loss

## Current Limitations

- Small vocabulary (demo purposes)
- Limited training data
- Basic preprocessing

## Development

- Type check: `npx tsc --noEmit`
- Build: `npm run build`
- Preview: `npm run preview`