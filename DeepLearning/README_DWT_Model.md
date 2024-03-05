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

### Wavelet Types

- **Haar:** Known for its simplicity and discontinuity, resembling a step function. Ideal for quick experiments and edge detection.

- **Daubechies:** Offers compactly supported orthonormal wavelets, suitable for practical discrete wavelet analysis. Identified by dbN, where N is the order.

- **Biorthogonal:** Features linear phase properties, making it suitable for signal and image reconstruction tasks.

- **Coiflets:** Designed by Daubechies, these wavelets have both wavelet and scaling functions with a certain number of vanishing moments, supporting signal smoothness.

- **Symlets:** Nearly symmetrical, these are modified versions of Daubechies wavelets, aiming for less asymmetry.

- **Morlet:** Known for not having a scaling function, explicit in form, and useful for frequency analysis.

- **Mexican Hat:** Also known as the Ricker wavelet, derived from the second derivative of the Gaussian function, useful for 2D and 3D data analysis.

- **Meyer:** Defined in the frequency domain, suited for smooth transitions between pass and stop bands.
