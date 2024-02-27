# Project Checklist

## Preparation

- [ ] Review literature on DWT and its application in neural network compression.
- [ ] Select a simple deep learning model architecture for initial experiments.
- [ ] Choose an appropriate dataset from Hugging Face for the task.

## Experimentation

- [x] Train the selected model on the chosen dataset to establish a baseline performance metric.
  - Using the MNIST keras data set for the intial traning of the traditional NN and DWT models.
- [x] Apply DWT to the model's weight matrices to increase sparsity.
  - See DWT_model.py under the Deep learning folder.
- [x] Retrain the model post-DWT application and compare performance to the baseline.
  - See the model_tester_main.py
  - [ ] Evaluate classification accuracy before and after DWT application, and save charts/images of results.
  - [ ] Assess model size reduction and efficiency improvements.

## Analysis

- [ ] Document the comparative performance in a structured manner.
- [ ] Use visual aids to demonstrate the impact of DWT on model performance and size.
- [ ] Analyze the results to determine the effectiveness of DWT in neural network compression.

## Iteration and Refinement

- [ ] Based on analysis, refine the DWT application process for optimization.
- [ ] Consider adjustments in the thresholding process for better sparsity without significant loss in performance.
- [ ] Explore different wavelets or transformation techniques as necessary.

## Documentation and Sharing

- [ ] Update the project README with findings and methodology.
- [ ] Prepare visual documentation to share with peers and advisors for feedback.
- [ ] Outline next steps based on project outcomes and feedback received.
