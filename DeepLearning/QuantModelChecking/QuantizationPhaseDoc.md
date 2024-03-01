# Quantization Phase Documentation

## Introduction

This document outlines the quantization phase of our neural network compression experiment. The aim is to reduce the model size through Discrete Wavelet Transform (DWT) and quantization without significantly impacting the model's performance. This phase is critical for deploying models on mobile and edge devices where memory is limited.

## Quantization Objectives

- **Determine the Impact of Quantization**: Assess how various levels of quantization affect the accuracy of neural network models.
- **Identify Critical Reduction Points**: Find the percentage of parameter reduction at which the model's performance sharply declines.
- **Performance Analysis**: Evaluate how the model's accuracy changes after each incremental reduction in parameter size.

## Experimental Checklist

- [ ] Train baseline models on the MNIST dataset (Model 1) and compare against a model with initial DWT application (Model 2).
- [ ] Apply quantization on Model 2 with a set threshold, and measure the parameter size reduction.
- [ ] Incrementally increase the strength of quantization and record the model's accuracy at each step.
- [ ] Determine the critical point at which the quantization begins to severely affect the model's performance.
- [ ] Compare the performance change after specific percentages of parameter size reduction.

## Goals

- **Model Size Reduction**: Quantify how much the model size can be reduced using DWT and quantization.
- **Performance Retention**: Establish the threshold of quantization that maintains acceptable performance levels.
- **Visualization of Results**: Create charts to illustrate the relationship between size reduction and performance.
- **Extend to Advanced Models**: Apply these techniques to more complex models like BERT and ResNet for a broader understanding of the quantization effects.

## Results and Analysis

After conducting the experiments, the results will be analyzed to understand:

- The trade-off between model size and accuracy.
- The maximum compression achievable without substantial accuracy loss.
- The scalability of this approach to more complex model architectures.

### Conclusions and Future Work

The findings from this phase will provide insights into the practicality of model compression techniques and will serve as a foundation for further research into adaptive learning methods and their applications in real-world scenarios.

#### Raw text for quant section

"""
model 1 and trained with mnist.
model 2 but quantized with dwt.
now what about the paramater size?
what is the paramater size?
with the mnist,
now have a threshold for the quantization, give this,
let's quantize more on the param sizes,
repeat this and increase the strength of the reduction
then find the point where the threshold pushes

where does the reduction threashold push the accuracy off the cliff.
evaluate the size and contnue to explore the

param size vs the performance,
find any critical points of % reduction.

what is the performance chagne after x% change?
accuracy, ...ect.

now we can measure the performnce of proof of concept,
the take on the BERT, or ResNet-pretrained
then do this on the text and image.

how much size reduction can we do before we end up reducing to much to not enough?
DWT matrix reduction, and reducing more, reconstruct back, and reconstruct the size.
then performance test.

relation between performnace and size reduction.
then far away is the adaptive learning process.'
put this in a chart, make the
PURPOSE: and then we can improve the vision then reduce the size for mobile/edge devices.

"""

Remove the above - these were just perliminary notes.
