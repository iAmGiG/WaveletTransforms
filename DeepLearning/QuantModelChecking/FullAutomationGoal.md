# Automation Pipeline for Model Optimization and Evaluation

This document outlines an automation pipeline to streamline the process from adjusting Discrete Wavelet Transform (DWT) model parameters, through TensorFlow Lite quantization, to evaluating quantized models.

## Steps in the Pipeline

1. **DWT Parameter Adjustment**:
   - Automate the adjustment of DWT parameters.
   - Train models with these parameters and log the results.

2. **Model Quantization**:
   - Apply TensorFlow Lite quantization on the trained models.
   - Save the quantized models for evaluation.

3. **Quantized Model Evaluation**:
   - Standardize evaluation of quantized models on metrics like accuracy and inference time.
   - Log these metrics for analysis.

4. **Visualization**:
   - Use MATLAB or Matplotlib to generate visual representations of the performance metrics at each stage.
   - These visuals aid in identifying optimal parameter settings.

## Implementation Suggestions

- **Scripting and Automation**: Use scripting languages like Python to manage parameter changes, initiate model training, quantization, and evaluation.
- **Logging and Tracking**: Implement a database or structured logging system to efficiently track experiments, parameter settings, and their corresponding outcomes.
- **Visualization Tools**: Integrate with MATLAB or Python's visualization libraries for generating insightful charts and graphs to illustrate the relationship between model compression and performance metrics.

## Goal

The ultimate objective is to build a comprehensive chart that maps the optimal parameters by analyzing the relationship between parameter size reduction and model performance, thereby identifying the most efficient models for deployment on edge devices.
