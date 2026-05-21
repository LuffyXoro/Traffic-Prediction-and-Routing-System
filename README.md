# AI-Based Traffic Prediction and Routing System

A deep learning-based smart traffic management system that predicts traffic congestion levels and provides intelligent alternative route suggestions using Bi-LSTM networks and real-time routing APIs.

## Overview

This project focuses on urban traffic congestion prediction using temporal deep learning models and integrates congestion-aware intelligent routing through an interactive Streamlit application.

The system analyzes historical traffic patterns using features such as:
- Traffic density
- Average speed
- Weather conditions
- Time of day
- Road type
- Rush hour indicators

A Bidirectional LSTM (Bi-LSTM) model is used to capture sequential traffic behavior and predict congestion levels as:
- Low
- Moderate
- High

The application also visualizes alternative traffic-aware routes using OpenRouteService and Folium maps.

---

## Features

- Traffic congestion prediction using Bi-LSTM
- Intelligent route suggestion system
- Real-time interactive Streamlit dashboard
- Temporal sequence modeling for traffic forecasting
- Feature engineering with rolling averages and cyclical encoding
- Interactive map visualization using Folium
- Multi-city traffic dataset support

---

## Tech Stack

- Python
- TensorFlow / Keras
- Bi-LSTM
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Folium
- OpenRouteService API

---

## Dataset

The project uses a historical traffic dataset containing:
- Traffic density
- Average speed
- Weather conditions
- Road type
- Rush hour indicators
- Geographic information
- Congestion labels

Target classes:
- Low Congestion
- Moderate Congestion
- High Congestion

---

## Model Architecture

```text
Input Sequence (30 timesteps)
        ↓
Bi-LSTM Layer (128 units)
        ↓
Dropout (0.3)
        ↓
Bi-LSTM Layer (64 units)
        ↓
Dense Layer
        ↓
Softmax Output (3 classes)

The Bi-LSTM model outperformed the Logistic Regression baseline across all evaluation metrics.

## Documentation

For detailed implementation, architecture, experimental results, and technical explanations, please refer to the [Project Report](./AI_Based_Traffic_Prediction_and_Routing_System_Report.pdf).