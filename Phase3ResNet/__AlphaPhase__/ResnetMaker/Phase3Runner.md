# Phase 3 Runner assistant

Helps with running the

## Contents

- `DWTResNetPruning.py`: Python script for pruning neural network models.
- `TestingScript.py`: Python script for testing the pruned models.
- `runner.py`: Runner script that orchestrates the pruning and testing processes.
- `config.json`: Configuration file containing parameters for the runner script.
- `README.md`: This documentation file.

## Configuration

The runner script is configured via the `config.json` file. Below is the structure and explanation of each parameter:

```json
{
  "model_directory": "./models",
  "log_directory": "./logs",
  "selective_prune_directory": "./prune_output/selective",
  "random_prune_directory": "./prune_output/random",
  "wavelet_type": "haar",
  "level": 1,
  "threshold": 0.5,
  "default_model_path": "./models/default_model.h5",
  "default_log_path": "./logs/default_log.csv"
}
```

## Parameters

- **model_directory**: The directory where the model files (.h5) are stored.
- **log_directory**: The directory where logs and result CSV files are stored.
- **wavelet_type**: The type of wavelet used for the DWT. Default is 'haar'.
- **level**: The level of decomposition in the DWT process. Default is 1.
- **threshold**: The threshold value used for pruning. Default is 0.5.
- **default_model_path**: Fallback path to the model file if no recent file is found.
- **default_log_path**: Fallback path to the log file if no recent file is found.

## Notes

- Ensure that all paths in the config.json file are correctly set according to your directory structure.
- The runner script requires absl-py for handling flags and subprocess for executing other Python scripts.

#### Note

- to actually get the hugging face models to run you'll have to learn about them first. I have made a mistake of not doing this.
- we, use the various HF libraries to actually work with the model directly, feels obvrious now, but took way to much time to get this understanding in.
