# Project Overview

## Objective

The core objective of this initial project is to explore the impact of applying Discrete Wavelet Transform (DWT) on the performance of neural networks, focusing on model compression and efficiency. This exploration will be grounded in a comparative study involving a simple neural network, before and after the application of DWT.

## Approach

1. **Model Selection:** Begin with a simple deep learning model, such as a perceptron or a basic multi-layer neural network, to serve as the baseline for experimentation.
2. **Dataset:** Utilize a straightforward dataset from Hugging Face for training purposes, ensuring the task is manageable yet meaningful for performance analysis.
3. **Wavelet Transformation Application:** Apply DWT to the neural network's weight matrices, aiming to increase sparsity and reduce the model's size.
4. **Performance Evaluation:** Conduct a before-and-after analysis, comparing the model's performance with and without wavelet transformation applied. This includes evaluating classification accuracy and model size reduction.
5. **Iterative Training:** With the wavelet-transformed (and thereby reduced) model, retrain on the same dataset to observe any changes in performance, assessing whether the model's efficiency or accuracy has improved or worsened.

## Expected Outcomes

- A proof of concept that demonstrates the feasibility and potential benefits of applying DWT to neural network model compression.
- Insight into how wavelet transformation affects neural network performance, specifically regarding classification tasks and model size.
- Data-driven decisions on the applicability of DWT for larger-scale models like BERT in future phases of the project.

## Visual Documentation

- Include visual comparisons (e.g., performance graphs, model size charts) to illustrate the impact of DWT on the neural network.
- Utilize visual tools to enhance the presentation and understanding of the project's outcomes.
