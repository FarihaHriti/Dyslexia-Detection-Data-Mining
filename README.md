# Generalizable Dyslexia Detection

> [!IMPORTANT]
> **Data Access**: [Download Final Datasets from Google Drive](https://drive.google.com/drive/folders/1BILJ5oZiVZWOjKHOcpRDoEmAE-tLMQtP?usp=drive_link)

This project implements a machine learning framework to detect dyslexia using eye-movement data and evaluates its generalization across different recording setups and child demographics.

## Research Goal
To answer the question: *"Can machine-learning models trained on reading data generalize across different recording setups and child age groups?"*

---

## Methodology (The 2 Experiments)

We implemented two specific experiments to validate this:

### 1. Experiment I: Intra-Dataset Baseline
*   **Goal**: Establish the baseline and optimized accuracy.
*   **Data**: **ETDD70** (Children, 9-10y).
*   **Models Evaluated**: **Random Forest**, **SVM (SVC)**, and **XGBoost**.
*   **Method**: Train and Test on ETDD70 (80/20 split).
*   **Labeling**: Linked to ground-truth labels via `dyslexia_class_label.csv`.

### 2. Experiment II: Cross-Dataset Generalization (The "Hard Test")
*   **Goal**: Test robustness against device/demographic shifts.
*   **Data**: Train on **ETDD70** -> Test on **Kronoberg** (Children, 2nd Grade, ~8y).
*   **Models Evaluated**: Zero-shot transfer using **Random Forest**, **SVM**, and **XGBoost**.
*   **Method**: Zero-Shot Transfer (Model sees Kronoberg data for the first time during testing).

---

## Performance Summary

The following table summarizes the comparative performance of our evaluated architectures across both experiments, showing the baseline scores alongside our optimized scores:

| Model Architecture | Exp I Baseline | Exp I Optimized | Exp II Baseline | Exp II Optimized | Key Observation / Optimization Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVM** | 81.8% | **90.9%** | 52.4% | **73.5%** | Switched to linear kernel for Exp I. Applied `MinMaxScaler` and optimized RBF parameters (C=0.5, gamma=0.1) for Exp II. |
| **Random Forest** | 81.8% | **81.8%** | 57.8% | **76.8%** | Applied `PowerTransformer` scaling and optimized trees (n=200, max_depth=5) for cross-dataset generalization. |
| **XGBoost** | 63.6% | **72.7%** | 56.8% | **79.5%** | Scaled with `PowerTransformer` and regularized hyperparameters (n=50, learning_rate=0.05, subsample=0.8) to mitigate domain shift. |

---

## Visual Analytics

Detailed visual reports are generated to analyze model performance across both experiments:

1. **Accuracy Comparison**: `multi_model_accuracy.png` - Visual bar chart of accuracy across all models.
2. **F1-Score Analysis**: `multi_model_f1_score.png` - Breakdown of classification performance for the Dyslexic group.
3. **Confusion Matrix Grid**: `multi_model_cm_grid.png` - Heatmaps for Experiment II showing true vs. predicted counts for all models.

---

### Technical Breakdown per Model

#### 1. SVM (Support Vector Machine)
*   **Exp I**: Linear kernel with `StandardScaler` provides an extremely clean hyperplane division, reaching **90.9%** accuracy.
*   **Exp II**: Extremely sensitive to raw gaze coordinate offsets. Applying `MinMaxScaler` bound all features within [0, 1], and tuning RBF kernel parameters raised accuracy to **73.5%**.

#### 2. Random Forest (Bagging)
*   **Exp I**: Strong baseline performance of **81.8%**.
*   **Exp II**: Susceptible to distribution mismatch. Applying `PowerTransformer` (which transforms feature distributions to be Gaussian-like) stabilized the tree split thresholds, lifting zero-shot accuracy to **76.8%**.

#### 3. XGBoost (Gradient Boosting)
*   **Exp I**: Achieved **72.7%** accuracy with tuned regularization parameters.
*   **Exp II**: The top performer at **79.5%**. Scaling the data using `PowerTransformer` combined with shallow trees (max_depth=3) and a small learning rate (0.05) prevented overfitting to the training domain and successfully generalized to the Swedish Kronoberg dataset.

---

## Project Structure

*   `src/data_loader.py`: Handles data loading and extraction.
    *   **Unified Features**: Calculates *Fixation Duration*, *Saccade Amplitude*, and Counts for all datasets.
    *   **Implements I-VT Algorithm**: Converts raw gaze data (Kronoberg) into fixation/saccade events.
*   `src/train_model.py`: Runs the 2 experiments and prints results.
*   `src/visualize_results.py`: Generates visual plots for the training/testing accuracy, F1-score, and confusion matrices.
*   `datasets/`: Contains the source datasets (ETDD70, Kronoberg).
*   `main.tex`: LaTeX report source file formatted for single-column presentation (suitable for Overleaf).

---

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
