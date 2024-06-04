# Project Overview

## Objective

**The core objective** of this initial project is to explore the impact of applying Discrete Wavelet Transform (DWT) on the performance of neural networks, 
focusing on model compression and efficiency. This exploration will be grounded in a comparative study involving a simple neural network, 
before and after the application of DWT.

## Approach

1. **Model Selection:**
	- Initial Phase: Start with simple models such as Multi-Layer Perceptrons (MLP) using datasets like MNIST for initial testing and validation.
	- Current Focus: Use more complex models, such as ResNet-18, trained on datasets like CIFAR-10, to validate the scalability and robustness of the approach.
2. **Dataset:** 
	- Initial Phase: Utilize the MNIST dataset for training simple models.
	- Current Focus: Use the CIFAR-10 dataset for evaluating ResNet models, moving away from ImageNet-1k due to better manageability and meaningful results with CIFAR-10.
3. **Wavelet Transformation Application:**
	- Apply DWT to the neural network's weight matrices to increase sparsity and reduce the model's size.
	- Evaluate different wavelet types, decomposition levels, and threshold values to optimize the pruning process.
4. **Performance Evaluation:** 
	- Conduct a comparative analysis of models before and after applying DWT.
	- Evaluate key metrics including classification accuracy, model size, sparsity, loss, F1 score, and recall.
	- Perform iterative retraining on the wavelet-transformed models to assess changes in performance and efficiency.
5. **Iterative Training:** 
	- Retrain the wavelet-transformed (pruned) models on the same dataset to observe any changes in performance, efficiency, and accuracy.
	- Focus on achieving a balance between model size reduction and maintaining acceptable performance levels.

## Expected Outcomes

- **Proof of Concept:** Demonstrate the feasibility and benefits of applying DWT for neural network model compression.
- **Performance Insights:** Provide insights into how wavelet transformation affects neural network performance, particularly in terms of classification tasks and model size.
- **Scalability:** Establish data-driven decisions on the applicability of DWT for larger-scale models, setting the stage for future work involving models like BERT.
- **Edge Computing:** Enhance the potential for neural networks to be deployed on edge devices with reduced computational demands and improved processing capabilities.

## Visual Documentation

- **Comparative Analysis:** Include visual comparisons such as performance graphs and model size charts to illustrate the impact of DWT on neural networks.
- **Enhanced Presentation:** Use visual tools to enhance the understanding and presentation of the project's outcomes, making it accessible for both technical and non-technical audiences.

## Recent Developments:
- **Wavelet Pruning:** Continued development of a detailed methodology for applying wavelet-based pruning using PyWavelet.
- **Model Testing:** Transitioned to testing with CIFAR-10 dataset for more meaningful evaluation results.
- **Expanded Metrics:** Expanded the evaluation metrics to include precision, average inference time, and size comparisons, despite some challenges in obtaining significant insights from inference time measurements.