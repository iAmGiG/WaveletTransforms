# DWT Model Training Instructions

This document outlines how adjusting the batch size impacts the training of models utilizing Discrete Wavelet Transform (DWT) for weight compression and provides instructions for running the code with flags.
Where you call the model.fit() method under the train with dwt model.

## Impact of Batch Size on DWT Models

Adjusting the batch size affects the number of samples that are processed before the model's internal parameters are updated.

- **Batch Size Increase**: Leads to faster training but might reduce the model's ability to generalize from DWT-compressed weights due to smoother convergence and higher memory usage.
Raising Batch Size: Increasing the batch size will generally
    lead to faster epoch times since there are fewer updates to be made.
- **Batch Size Decrease**: Potentially improves the model's performance with DWT-compressed weights through better generalization, albeit with increased training time and lower memory demand.
Lowering Batch Size: Decreasing the batch size often
    leads to longer training times since the model's parameters are updated more frequently.

## Running the Code

To train the model with customized settings, use the following command-line flags:

- `--wavelet`: Specifies the type of wavelet for DWT (default: 'haar').
- `--batch_size`: Determines the batch size for training (e.g., 32).
- `--epochs`: Sets the number of training epochs (e.g., 10).
- `--save_dir`: Directory to save trained models (default: './DeepLearning/SavedDWTModels').

Example command:

```sh
python DWT_model.py --wavelet haar --batch_size 32 --epochs 10 --save_dir ./SavedModels
```

## PyWavelet

**Increasing the Level of Decomposition:**

The level of decomposition refers to the number of times the wavelet decomposition is applied to the input signal. Each level of decomposition splits the signal into approximation and detail coefficients. The approximation coefficients represent the low-frequency components, while the detail coefficients represent the high-frequency components.

By increasing the level of decomposition, you can extract more detailed frequency information from the signal. This can be beneficial for deep learning models, as it can help capture relevant features at different frequency scales.

However, it's important to note that increasing the decomposition level too much can lead to an excessive number of coefficients, which may not be necessary or could even introduce noise. Finding the optimal level of decomposition is often a trade-off between capturing enough relevant information and avoiding excessive complexity.

### Tools and adjusting the various levels

1. **Adjusting the Decomposition Level:** Increasing the level of decomposition allows the DWT to analyze the signal at finer resolutions, potentially enhancing compression by focusing on significant coefficients.

2. **Thresholding Wavelet Coefficients:** Implementing a more aggressive thresholding strategy, such as higher values for hard or soft thresholding, effectively zeros out smaller coefficients, increasing sparsity.
Applying thresholding to the wavelet coefficients can help in several ways:

    - **Denoising:** Thresholding can help remove or reduce the impact of noise present in the input signal, which can improve the quality of the features extracted for the deep learning model.
    - **Sparsity:** Thresholding can introduce sparsity in the wavelet coefficients, meaning that many coefficients will be set to zero or close to zero. This sparsity can be beneficial for deep learning models, as it can reduce the dimensionality of the input data and potentially improve the model's performance and interpretability.
    - **Feature Selection:** By applying thresholding, you are effectively selecting the most relevant wavelet coefficients, which can act as features for the deep learning model. This can help the model focus on the most important information and potentially improve its performance.

3. **Selecting Wavelets with Higher Vanishing Moments:** Some wavelets, like higher-order Daubechies, provide better compression for specific types of data due to their ability to capture more information in fewer coefficients.

### Wavelet Types

- **Haar:** Known for its simplicity and discontinuity, resembling a step function. Ideal for quick experiments and edge detection.

```py
#    --wavelet=harr
```

- **Daubechies:** Offers compactly supported orthonormal wavelets, suitable for practical discrete wavelet analysis. Identified by dbN, where N is the order.

```py
#   --wavelet=db1
#   --wavelet=db2
```

- **Biorthogonal:** Features linear phase properties, making it suitable for signal and image reconstruction tasks. (biorthogonal and reverse biorthogonal)

```py
#   --wavelet=bior1.3
#   --wavelet=rbio1.3
```

- **Coiflets:** Designed by Daubechies, these wavelets have both wavelet and scaling functions with a certain number of vanishing moments, supporting signal smoothness.

```py
#   --wavelet=coif1
```

- **Symlets:** Nearly symmetrical, these are modified versions of Daubechies wavelets, aiming for less asymmetry.

```py
#   --wavelet=sym2
```

- **Morlet:** Known for not having a scaling function, explicit in form, and useful for frequency analysis.

```py
#   --wavelet=morl
```

- **Mexican Hat:** Also known as the Ricker wavelet, derived from the second derivative of the Gaussian function, useful for 2D and 3D data analysis.

```py
#   --wavelet=mexh
```

- **Meyer:** Defined in the frequency domain, suited for smooth transitions between pass and stop bands.

```py
#   --wavelet=dmey
```

- **Gaussian Wavelets (gaus):** Primarily used in the analysis of functions with Gaussian behavior or for Gaussian derivatives in signal processing.

```py
#   --wavelet=gaus
```

- **Complex Gaussian Wavelets (cgau):** Useful for capturing information in signals where both amplitude and phase are important, used in complex signal analysis.

```py
#   --wavelet=cgau
```

- **Shannon Wavelets (shan):** Based on the Shannon or sinc function, ideal for applications requiring exact reconstruction of signals with finite bandwidth.

```py
#   --wavelet=shan
```

- **Frequency B-Spline Wavelets (fbsp):** Combines B-spline functions and Fourier transforms, useful for signal smoothing and approximation tasks.

```py
#   --wavelet=fbsp
```

- **Complex Morlet Wavelets (cmor):** A complex wavelet, ideal for time-frequency analysis where frequency localization is critical.

```py
#   --wavelet=cmor
```

#### future consideration

- **Hyperparameter Tuning:** While you have provided flags for adjusting the batch size, epochs, and wavelet type, you could consider implementing a more systematic approach to hyperparameter tuning. This could involve techniques like grid search, random search, or Bayesian optimization to find the optimal combination of hyperparameters for your model.

- **Cross-Validation:** To ensure the robustness and generalization capability of your model, you could implement cross-validation techniques, such as k-fold cross-validation. This would help you evaluate your model's performance on multiple splits of the data and provide a more reliable estimate of its generalization ability.

- **Early Stopping:** Implementing an early stopping mechanism during training could help prevent overfitting and potentially improve the model's performance. You could monitor the validation loss or accuracy and stop training when these metrics plateau or start to degrade.

- **Learning Rate Scheduling:** Experimenting with different learning rate schedules, such as step decay, exponential decay, or cyclical learning rates, could lead to faster convergence and potentially better performance.

- **Data Augmentation:** If your dataset is relatively small or lacks diversity, you could consider applying data augmentation techniques, such as rotation, flipping, or adding noise, to artificially increase the size and diversity of your training data.

- **Model Ensembling:** Instead of training a single model, you could explore ensemble techniques, such as bagging or boosting, to combine multiple DWT-compressed models and potentially improve overall performance.

- **Visualizations and Interpretability:** Adding visualizations and interpretability techniques, such as activation maps or saliency maps, could provide insights into how the DWT-compressed weights affect the model's behavior and decision-making process.

- **Regularization Techniques:** Exploring different regularization techniques, such as L1 or L2 regularization, dropout, or batch normalization, could help prevent overfitting and potentially improve the model's generalization ability.

- **Transfer Learning:** If you have access to pre-trained models or larger datasets, you could consider leveraging transfer learning techniques to initialize your model's weights with pre-trained values and fine-tune them on your specific task.

- **Distributed Training:** If you have access to multiple GPUs or a distributed computing environment, you could explore distributed training strategies to accelerate the training process and potentially handle larger batch sizes or more complex models.
