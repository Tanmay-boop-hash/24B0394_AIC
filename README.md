# Fashion-MNIST Classification using ResNet50

This project uses **transfer learning** with a pre-trained **ResNet50** convolutional neural network to classify images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) into 10 categories of clothing and accessories.

## Dataset

The dataset consists of:
- 60,000 training images
- 10,000 test images
- 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

Each image is a **28x28 grayscale image**.

---

## Model

We used **ResNet50**, a deep residual network pre-trained on ImageNet, and adapted it to work with Fashion-MNIST by:

- Resizing input images to 224x224 (to match ResNet’s input size)
- Converting grayscale to RGB
- Freezing base layers initially, then fine-tuning
- Adding custom dense output layers

---

## Results

- **Test accuracy:** 94%
- Visualizations showed the model correctly classifying most items with confidence
- Used early stopping and fine-tuning for better generalization

---

## Files

- `Fashion_MNIST_ResNet50_TransferLearning.ipynb` — Full code notebook
- `resnet50_fashion_mnist.keras` — Trained model file (270 MB)
  - [Download from Google Drive](https://drive.google.com/file/d/1aOiuV_fWZVmxNEgWEJB7kQPeMMe0AXMx/view?usp=sharing)

---

## Run It on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tK8qkGcEaGuP8MNCsc6Xgkdf7Bl5M-ec?usp=drive_link)

---



# RAG Chatbot with Groq API + Agentic Architecture

A context-aware PDF chatbot that uses **retrieval-augmented generation (RAG)** with **open-source LLMs via Groq**. Supports:
- FAISS-based semantic search
- Follow-up question rewriting
- Agent-based processing (extract → synthesize → answer)
- History-aware multi-turn conversations

---

## Features

- **PDF Ingestion**: Extracts and chunks readable content from PDFs.
- **Vector Search**: Uses `sentence-transformers` + `FAISS` for fast retrieval.
- **Open-Source LLMs**: Uses `llama3-8b-8192` or `mixtral` models via Groq API.
- **Agentic Pipeline**:
  - `InfoExtractionAgent` → pulls key facts
  - `SynthesisAgent` → organizes knowledge
  - `QueryAgent` → answers the user’s question
- **Follow-up Question Handling**: Rewrites vague questions using chat history.

---

## Demo (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10hx93LWFNTWwu1306h5x0Kv2IEYZnUju?usp=sharing)

---


