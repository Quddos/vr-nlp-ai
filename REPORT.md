# VR-NLP-AI Project Report

## Overview
This project extends a default Immersive Web SDK (IWSDK) application with a neural machine translation (NMT) module inspired by the paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau, Cho, and Bengio.

The implementation combines:
- a Python backend running a PyTorch sequence-to-sequence model with attention,
- a Flask API serving translation requests,
- an IWSDK frontend that exposes a VR UI panel for text input and translation output.

## What the Model Does
The model translates short English sentences into French. It is built using an encoder-decoder architecture with an attention mechanism.

### Encoder
- Converts input tokens into embeddings.
- Runs a bidirectional GRU over the source sentence.
- Outputs encoder hidden states for every source position and a final hidden representation.

### Attention
- Implements Bahdanau-style additive attention.
- Computes alignment scores between the current decoder state and each encoder output.
- Uses a softmax to produce attention weights over source tokens.
- Produces a weighted context vector that focuses the decoder on relevant source words.

### Decoder
- Converts the previously generated token and attention context into the next token prediction.
- Uses a GRU that receives the embedded previous token concatenated with the attention-weighted encoder context.
- Produces a probability distribution over the target vocabulary.

### Seq2Seq Training
- The model is trained with teacher forcing using cross-entropy loss.
- During training, the decoder is given the true previous target token with probability 0.5.
- The loss ignores padding tokens.

## What Was Implemented
The repository now contains:
- `app.py`: the full PyTorch NMT model plus Flask API.
- `REPORT.md`: this report explaining the model and integration.
- `requirements.txt`: Python dependency declarations.
- `package.json` update: added `python-server` script.
- `ui/welcome.uikitml`: updated to include translation input/output controls.
- `src/panel.ts`: frontend logic to call the translation API from the VR UI.

## How It Was Implemented
### Backend Implementation
`app.py` defines:
- `Encoder` class: bidirectional GRU encoder.
- `Attention` class: additive attention network.
- `Decoder` class: GRU decoder with context concatenation.
- `Seq2Seq` class: end-to-end encoder-decoder training loop.
- `Vocab` class: very small token vocabulary for demo.
- `TranslationDataset`: wraps sample sentence pairs.
- `train`, `evaluate`, and `translate_sentence` functions.

The backend also includes:
- Flask server with `/translate` and `/health` endpoints.
- Model save/load logic for `nmt_model.pth`.
- A small demonstration dataset built directly in code.

### Frontend / VR Integration
The VR frontend integration uses the existing IWSDK panel system.
- `ui/welcome.uikitml` now includes an input field, output field, and a Translate button.
- `src/panel.ts` adds event listeners that call the Flask API and populate the translation result.
- The existing XR enter/exit button remains intact.

This integration keeps the translation feature within the same panel UI used by the IWSDK app.

## How It Works in VR
In the VR application:
- The user opens the panel UI in the immersive environment.
- They type or paste English text into the input panel.
- They press the `Translate` button.
- The frontend sends a POST request to `http://localhost:5000/translate`.
- The Python backend translates the sentence and returns the target French text.
- The translated text appears in the output box.

Because the UI is rendered through IWSDK, users can interact with the text controls inside the immersive scene.

## Impact and Problem Solved
This project demonstrates how to integrate NLP into immersive applications.

### Problems addressed
- Adds natural language understanding and translation capability to an IWSDK experience.
- Connects a VR frontend to a machine learning backend.
- Provides a working example of combining real-time AI services with spatial UI.

### Impact
- Enables immersive translation demos in WebXR.
- Serves as a starting point for richer multilingual VR interfaces.
- Shows how to attach a backend ML service to an IWSDK panel.

## Technology Stack
### Backend
- Python 3
- PyTorch
- Flask
- NumPy

### Frontend
- TypeScript / JavaScript
- Immersive Web SDK (`@iwsdk/core`)
- Vite development server
- UIKitML for VR panel UI

### Model Architecture
- Encoder-decoder sequence model
- Bidirectional GRU encoder
- GRU decoder
- Bahdanau-style attention

## Dataset Source
This project currently uses a tiny, hard-coded demo dataset inside `app.py`:
- English sentences: `hello world`, `how are you`, `good morning`
- French sentences: `bonjour monde`, `comment allez vous`, `bonjour`

This is not a production dataset. It is only a minimal example dataset for demonstration and integration validation.

## Accuracy and Limitations
### Accuracy
- The current demo dataset is extremely small.
- Accuracy is effectively demonstration-level, not production-level.
- In tests, the model can translate the small training examples correctly after training.

### Limitations
- Vocabulary is tiny and fixed.
- No real text preprocessing beyond token splitting.
- No real dataset loading or large-scale training.
- The translation API returns `<unk>` for unknown tokens outside the small vocabulary.
- The model is not robust for general English-to-French translation.

## How to Run and Test
### Requirements
- Node.js
- npm
- Python 3
- PyTorch
- Flask

### Install dependencies
```bash
pip install torch numpy flask
npm install
```

### Start the Python translation server
```bash
python app.py
```
This trains the model if `nmt_model.pth` does not exist, then starts Flask on `http://localhost:5000`.

### Start the IWSDK app
```bash
npm run dev
```
Then open `https://localhost:8081/`.

### Test translation manually
Use the VR panel input, or test the API directly with curl:
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"sentence":"hello world"}'
```

### Expected result
The demo should return a French translation such as:
```json
{"translation":"bonjour monde"}
```

## API Connections
Yes, the VR frontend connects to a local Flask API.

### Endpoints
- `GET /health`
  - Returns `{"status":"ok"}`
- `POST /translate`
  - Request body: `{"sentence": "..."}`
  - Response body: `{"translation": "..."}`

### Integration
The IWSDK panel sends the translation request via fetch to `http://localhost:5000/translate`.
The backend returns JSON and the frontend updates the VR UI panel.

## Notes and Recommendations
This implementation is a working prototype. For a production-grade system:
- Replace the demo dataset with a real bilingual corpus.
- Add tokenizer / subword encoding.
- Use a larger vocabulary or pretrained translation model.
- Move the backend to a proper server host and secure the API.
- Improve the UI for more realistic VR text entry.

## Summary
The project successfully integrates a neural machine translation model into an IWSDK application.
It demonstrates how to connect immersive UI controls to a Python ML backend through a simple REST API.
It is a clear prototype of multimodal VR + NLP interaction, with room to expand toward real translation datasets and robust service deployment.
