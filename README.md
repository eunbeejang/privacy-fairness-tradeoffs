## privacy-fairness-tradeoffs


# Arguments options in `main.py`
```
BANK Dataset

optional arguments:
  -h, --help            show this help message and exit
  -b B, --batch-size B  input batch size for training (default: 128)
  --test-batch-size TB  input batch size for testing (default: 1024)
  -n N, --epochs N      number of epochs to train (default: 20)
  -s SPLIT, --split SPLIT
                        test split ratio (default: .1)
  -r N_RUNS, --n-runs N_RUNS
                        number of runs to average on (default: 1)
  --lr LR               learning rate (default: .1)
  --sigma S             Noise multiplier (default 1.0)
  -c C, --max-per-sample-grad_norm C
                        Clip per-sample gradients to this norm (default 1.0)
  --delta D             Target delta (default: 1e-5)
  --device DEVICE       GPU ID for this process (default: 'cuda')
  --save-model          Save the trained model (default: false)
  --disable-dp          Disable privacy training and just train with vanilla SGD
  --data-root DATA_ROOT
                        Where BANK_DATASET is/will be stored
```




# Method 1: Original model with original data
```
python main.py --disable-dp
```

# Method 2: Original model with differentially private data
```
# create synthetic data first
python dp_data_synth.py

# Run model
python main.py --data-root ./bank-data/synth/random_mode/sythetic_data.csv
```

# Method 3: Noisy SGD
```
python main.py
```
