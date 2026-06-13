# Generalizable Dyslexia Detection

> [!IMPORTANT]
> **Data Access**: [Download Final Datasets from Google Drive](https://drive.google.com/drive/folders/1BILJ5oZiVZWOjKHOcpRDoEmAE-tLMQtP?usp=drive_link)

This project implements a machine learning framework to detect dyslexia using eye-movement data and evaluates its generalization across different recording setups and child demographics.

## Research Goal
To answer the question: *"Can machine-learning models trained on reading data generalize across different recording setups and child age groups?"*

## Methodology (The 2 Experiments)

We implemented two specific experiments to validate this:

### 1. Experiment I: Intra-Dataset Baseline
*   **Goal**: Establish the baseline accuracy.
*   **Data**: **ETDD70** (Children, 9-10y).
*   **Models Evaluated**: **Random Forest**, **SVM (SVC)**, and **XGBoost**.
*   **Method**: Train and Test on ETDD70 (80/20 split).
*   **Note**: Uses label mapping for classification (Subject ID < 1100 = Control, Subject ID >= 1100 = Dyslexic).

### 2. Experiment II: Cross-Dataset Generalization (The "Hard Test")
*   **Goal**: Test robustness against device/demographic shifts.
*   **Data**: Train on **ETDD70** -> Test on **Kronoberg** (Children, 2nd Grade, ~8y).
*   **Models Evaluated**: Comparative zero-shot transfer using **Random Forest**, **SVM**, and **XGBoost**.
*   **Method**: Zero-Shot Transfer (Model sees Kronoberg data for the first time during testing).

---

## Performance Summary

The following table summarizes the comparative performance of our evaluated architectures across both Supervised Experiments (Exp I: Intra-Dataset, Exp II: Cross-Dataset):

| Model Architecture | Exp I Accuracy | Exp II Accuracy | Key Observation |
| :--- | :--- | :--- | :--- |
| **SVM (RBF)** | 63.6% | 52.4% | Moderate intra-dataset baseline, but extremely sensitive to domain shifts. |
| **Random Forest** | 63.6% | 57.8% | Moderate performance; slightly more robust than SVM in zero-shot Kronoberg. |
| **XGBoost** | **72.7%** | **63.2%** | **Most Stable.** Maintains consistent performance across differing datasets and hardware. |

## Visual Analytics

Detailed visual reports are generated to analyze model performance across both experiments:

1. **Accuracy Comparison**: `multi_model_accuracy.png` - Visual bar chart of accuracy across all models.
2. **F1-Score Analysis**: `multi_model_f1_score.png` - Breakdown of classification performance for the Dyslexic group.
3. **Confusion Matrix Grid**: `multi_model_cm_grid.png` - Heatmaps for Experiment II showing true vs. predicted counts for all models.

---

### Technical Breakdown per Model

#### 1. Random Forest (Bagging)
- **Exp I**: Captures specific patterns within the ETDD70 dataset.
- **Exp II**: Suffered significantly from "Domain Shift." Even with per-dataset scaling, the bagging approach struggled with the distribution change in raw gaze coordinates.

#### 2. SVM (Support Vector Machine)
- **Exp I**: Achieved strong accuracy on the training distribution.
- **Exp II**: Suffered under domain shift as the rigid boundary struggled to generalize without explicit domain normalization.

#### 3. XGBoost (Gradient Boosting)
- **Exp I**: Top performer. The iterative boosting process minimized error successfully.
- **Exp II**: Successfully crossed the 60% threshold. The combination of **L1/L2 regularization** and **Quantile Mapping** allowed the model to focus on relative ratios rather than absolute values, effectively mitigating hardware bias.

> [!TIP]
> **Quantile Mapping** was crucial for this project. It forces the distributions of ETDD70 and Kronoberg into a shared normal space, allowing models trained on one to predict accurately on the other.

## Project Structure

*   `src/data_loader.py`: Handles data loading and extraction.
    *   **Unified Features**: Calculates *Fixation Duration*, *Saccade Amplitude*, and Counts for all datasets.
    *   **Implements I-VT Algorithm**: Converts raw gaze data (Kronoberg) into fixation/saccade events.
*   `src/train_model.py`: Runs the 2 experiments and prints results.
*   `src/visualize_results.py`: Generates visual plots for the training/testing accuracy, F1-score, and confusion matrices.
*   `datasets/`: Contains the 2 source datasets (ETDD70, Kronoberg).

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install pandas scikit-learn openpyxl xlrd matplotlib seaborn xgboost
    ```

2.  **Run the Experiments**:
    ```bash
    python src/train_model.py
    ```

3.  **Generate Visual Charts**:
    ```bash
    python src/visualize_results.py
    ```

4.  **View Results**:
    *   Check the terminal for Accuracy reports on Experiment I and II.
    *   Open `multi_model_accuracy.png` and other generated images for visual comparisons.
