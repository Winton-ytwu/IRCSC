# Paper Implementation: Task-Oriented Semantic Communication with Importance-Aware Rate Control (IRCSC)

### Overview

This project is an implementation of the paper **"Task-Oriented Semantic Communication with Importance-Aware Rate Control" (arXiv:2504.20441v1)**. 
The core idea is to address the challenge of semantic communication systems that use a fixed transmission rate in dynamic channel environments. Such systems can be inefficient.

This work introduces the **Importance-Aware Rate Control Semantic Communication (IRCSC)** scheme. This scheme dynamically adjusts the transmission rate based on two key factors:
1.  **Channel Conditions** (e.g., Signal-to-Noise Ratio, SNR).
2.  **Semantic Importance** of the features being transmitted.

The goal is to achieve a controllable trade-off between task performance (e.g., classification accuracy) and transmission rate, ensuring high efficiency and robustness.

### Key Features & Contributions

This implementation reproduces the main contributions of the paper:
* **Contribution-based Importance Analyzer**: An efficient module to evaluate the importance of different feature channels without significant computational overhead.
* **Semantic Transmission Integrity Index (STII)**: A novel metric that combines semantic importance and Bit Error Rate (BER) to quantify the quality of the semantic transmission.
* **Adaptive Rate Control**: A mechanism that uses the STII to dynamically select the minimum number of features to transmit, satisfying a given performance threshold under current channel conditions.

### System Architecture

The system consists of a transmitter, a wireless channel, and a receiver. The key innovation lies within the transmitter, which intelligently selects features before transmission.

<img width="616" height="211" alt="image" src="https://github.com/user-attachments/assets/6cde498c-9c8a-44cb-860f-ffc1e8818c8a" />


### File Structure

The repository is organized as follows:

| File/Folder | Description |
| :--- | :--- |
| `data/` | Directory to store datasets (e.g., CIFAR-10). |
| `models/` | Directory to save trained model weights (`.pth` files). |
| `results/` | Directory to save experiment outputs like JSON data and plots. |
| `model.py` | Defines the `JSCC_Classifier` neural network architecture. |
| `utils.py` | Contains helper functions for channel simulation, importance calculation, STII, etc. |
| `data_load.py` | Scripts for loading CIFAR-10 and other datasets. |
| `train.py` | Script to train the baseline TD-JSCC model. |
| `create_mapping_data.py` | Generates the data points for the STII-vs-Accuracy mapping function (Algorithm 1). |
| `fit_curve.py` | Fits a rational function to the mapping data to find the parameters of `φ(η)`. |
| `run_final_evaluation.py` | Runs the core comparison between IRCSC and TD-JSCC across all SNRs. |
| `plot_final_results.py` | Generates the final plots comparing the performance and transmission rates. |
| `requirements.txt` | A list of Python packages required to run the project. |
| `run_ablation_study.py`|Runs an ablation study to verify the effectiveness of the importance-aware feature selection. |

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Datasets:**
    The `data_load.py` script will automatically download the CIFAR-10 dataset into the `./data` directory when first run.

### Workflow & Usage

To reproduce the results from the paper, follow these steps in order:

**Step 1: Train the Baseline TD-JSCC Model**
This step trains the model with a fixed compression ratio and saves the weights. This model will be used for all subsequent steps.
```bash
python train.py --compression_ratio 0.0833 --snr_db 0 --epochs 200
```

**Step 2: Generate STII-Accuracy Mapping Data (Algorithm 1)**
This script simulates the transmission under various conditions to collect data points that map STII values to task accuracy.
```bash
python create_mapping_data.py --model_path ./models/td_jscc_cifar10_snr0_awgn_kn0.0833.pth --k_value 4
```

**Step 3: Fit the Mapping Function Curve**
This script takes the generated `mapping_data.json` and fits the rational function from the paper (Eq. 11) to find the parameters.
```bash
python fit_curve.py
```

**Step 4: Run the Final Evaluation (IRCSC vs. TD-JSCC)**
This is the main evaluation script. It compares the adaptive IRCSC scheme against the fixed-rate TD-JSCC baseline across a range of SNRs.
```bash
python run_final_evaluation.py --model_path ./models/td_jscc_cifar10_snr0_awgn_kn0.0833.pth --k_value 4
```

**Step 5: Plot the Final Results**
This script visualizes the output from the previous step, generating the final comparison plots for accuracy and transmission rate.
```bash
python plot_final_results.py --result_file ./results/final_evaluation_results_k4.json --k_value 4
```


