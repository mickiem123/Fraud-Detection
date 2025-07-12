
# Transaction Fraud Detection with a Bagging CNN Ensemble
![alt text](https://img.shields.io/badge/Python-3.9-blue.svg)
![alt text](https://img.shields.io/badge/PyTorch-1.12-orange.svg)
![alt text](https://img.shields.io/badge/scikit--learn-1.1.2-159953.svg)
![alt text](https://img.shields.io/badge/FastAPI-0.85-009688.svg)
![alt text](https://img.shields.io/badge/Docker-20.10-blue.svg)
![alt text](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)


## Introduction

This project presents an end-to-end machine learning system for high-performance transaction fraud detection. Although developed on a synthetic dataset, the methodology emphasizes real-world constraints—such as severe class imbalance, imperfect information and the need for a deployable, low-latency model—to simulate a production-ready solution.

This project includes 2 model, a Feed Forward Neural Network (FFNN) to serve as a baseline model for comparision, and a Bagging Convolutional Neural Neural (Bagging CNN) as the main model.

**Key Features**

- Advanced Pattern Recognition: Utilizes a 1D Convolutional Neural Network (CNN) to automatically extract complex, contextual patterns from transaction feature data, moving beyond traditional models.

- Ensemble for Robustness: Implements a Bagging ensemble of CNNs to reduce model variance and improve generalization, resulting in a more stable and reliable predictive system.

- End-to-End Deployment: The final model is fully containerized with Docker and served via a REST API using FastAPI, demonstrating a complete "model-to-production" workflow deployed on Render.

- Business-Centric Metrics: Performance is optimized for high Recall to catch most fraud transaction while also ensure a high precision to avoid flagging to much non-fraud transactions, to not disturb customer. The model was also evaluated on Precision@Top-K, a practical metric for operational teams with limited review capacity.
**Tech Stack**

- Backend: Python, FastAPI, Gunicorn

- ML/Data Science: PyTorch, PyTorch Lightning, Scikit-learn, Pandas, NumPy

- Deployment: Docker, Render


```text
+----------------+      +-------------------+      +-----------------+      +---------------------+
|   Raw Data     |----->|  Data Processing  |----->|  Model Training |----->|   Saved Model     |
| (.csv file)    |      | (Pandas, Sklearn) |      | (PyTorch/CNN)   |      | (.pth file)       |
+----------------+      +-------------------+      +-----------------+      +---------------------+
                                                                                   |
                                                                                   | (loaded by API)
                                                                                   v
                                                   +-------------------+      +-------------------+
                                                   |  Docker Container |----->|    FastAPI        |
                                                   |  (API Environment)|      | (Prediction Logic)|
                                                   +-------------------+      +-------------------+
```




## Installation & Local Setup

This project has two distinct setup paths depending on your goal: a **full development setup** to run the training and experimentation code, and a **lightweight API-only setup** to run the deployed application. This separation is a best practice to ensure the production environment is lean and secure.

### Path 1: Full Development Environment (For Re-training the Model)

This path installs all packages required for data exploration, model training, and running the API.

- **Clone the Repository:**

    ```bash
    git clone https://github.com/mickiem123/Fraud-Detection
    cd Fraud-Detection
    ```
- **Create and Activate a Virtual Environment:**
    ```bash
    # Create a virtual environment
    python -m venv fraud_detection_venv

    # Activate it
    # On Windows:
    fraud_detection_venv\Scripts\activate
    # On macOS/Linux:
    source fraud_detection_venv/bin/activate
    ```
- **Install All Dependencies:**
    This uses the main `requirements.txt` file which contains both training and deployment packages.
    ```bash
    pip install -r requirements.txt
    ```

### Path 2: Lightweight API Environment (For Running the Deployed App Locally)

This path installs only the minimal packages needed to run the API, mirroring the Docker environment.

- **Follow Steps 1 & 2** from the Full Development setup above.
- **Install API-only Dependencies:**
    Note that we are using the `requirements.txt` file from *inside* the `fraud_detection_api` folder.
    ```bash
    pip install -r fraud_detection_api/requirements.txt
    ```


After setting up either environment, you can run the FastAPI server from the `fraud_detection_api` directory.

```bash
# First, change into the API directory
cd fraud_detection_api

# Run the server
uvicorn main:app --reload

# You can now access the API documentation at http://127.0.0.1:8000/docs

```

## Dataset

The project is built upon the **synthetic credit card fraud detection dataset** created by the Machine Learning Group at Université Libre de Bruxelles (ULB).

### 1. Data Understanding & Exploratory Data Analysis (EDA)

The foundation of the project is a thorough analysis of the dataset to uncover its underlying structure and identify patterns indicative of fraud. While the dataset is *synthetic*, I tried to implements standard and logical techniques that can also be applied to a real-world problem.

#### A. Dataset Overview

*   **Source:** `https://github.com/Fraud-Detection-Handbook/simulated-data-raw`
*   **Scope:** The dataset contains **1,754,155** transactions.
*   **The Imbalance Problem:** A critical challenge is the severe class imbalance, with only **14,681** fraudulent transactions (0.84% of the data).
*   **Target Variable:** The goal is to predict the `TX_FRAUD` column (1 for fraud, 0 for legitimate).
*   **Data Quality:** As a synthetic dataset, it is well-structured and required no significant data cleaning.

Below is a summary of the features:

| Feature             | Description                                | Data Type           | Range                         |
| :------------------ | :----------------------------------------- | :------------------ | :---------------------------- |
| `TRANSACTION_ID`    | Unique identifier for each transaction.    | Numerical (Integer) | 0 to ~1.75M                   |
| `TX_DATETIME`       | Timestamp of the transaction.              | Datetime            | Apr 2018 - Sep 2018           |
| `CUSTOMER_ID`       | Unique identifier for each customer.       | Numerical (Integer) | 1 to 5,000                    |
| `TERMINAL_ID`       | Unique identifier for each payment terminal. | Numerical (Integer) | 1 to 10,000                   |
| `TX_AMOUNT`         | The monetary value of the transaction.     | Numerical (Float)   | 0.00 to 2,628.00              |
| `TX_TIME_SECONDS`   | Seconds elapsed since the first transaction. | Numerical (Integer) | 0 to ~1.57e7                  |
| `TX_TIME_DAYS`      | Days elapsed since the first transaction.  | Numerical (Integer) | 0 to 182                      |
| `TX_FRAUD`          | Target variable: 1 if fraud, 0 otherwise.  | Boolean (0/1)       | 0 or 1                        |
| `TX_FRAUD_SCENARIO` | A code for the type of fraud scenario.     | Categorical         | 1, 2, or 3                    |

#### B. Key EDA Findings

A comprehensive EDA was conducted, leading to several critical insights that guided the project:

*   **Sequential Nature of Fraud:** A key finding is that fraudulent transactions often occur in rapid succession for the same `CUSTOMER_ID` or `TERMINAL_ID`. This insight is the primary motivation for using sequence-aware models.

*   **Temporal Patterns:** Fraudulent activities are disproportionately more likely to occur **during the night**, a common pattern suggesting exploitation of lower monitoring periods.

*   **Transaction Amount Discrepancies:** On average, fraudulent transactions have a significantly higher monetary value, though considerable overlap exists with normal transactions.

| Statistic       | Fraudulent Transactions | Normal Transactions |
| :-------------- | :---------------------- | :------------------ |
| **Mean Amount** | **131.17**              | 52.98               |
| Std Dev         | 154.49                  | 39.42               |
| Median (50%)    | 72.22                   | 44.49               |
| Max Amount      | 2,628.00                | 219.98              |

### 2. Feature Engineering: Crafting Predictive Signals

To provide the model with more predictive signals, several new features were engineered based on the EDA findings. The transformation logic is encapsulated within a modular **Pipeline Design Pattern**, making it easily extensible (`fraud_detection/src/components/feature_engineering.py`).

The core feature engineering strategy is summarized below:

| Original feature name | Transformed feature name(s)          | Transformation Logic                                          |
| :-------------------- | :----------------------------------- | :------------------------------------------------------------ |
| `TX_DATE_TIME`        | `TX_DURING_WEEKEND`                  | Binary flag (1) for weekend transactions.                     |
| `TX_DATE_TIME`        | `TX_DURING_NIGHT`                    | Binary flag (1) for night transactions (8pm-6am).             |
| `CUSTOMER_ID`         | `CUSTOMER_ID_NB_TX_NDAY_WINDOW`      | Customer's transaction count in rolling 1, 7, 30-day windows. |
| `CUSTOMER_ID`         | `CUSTOMER_ID_AVG_AMOUNT_NDAY_WINDOW` | Customer's average spend in rolling 1, 7, 30-day windows.     |
| `TERMINAL_ID`         | `TERMINAL_ID_NB_TX_NDAY_WINDOW`      | Terminal transaction count in delayed 1, 7, 30-day windows.   |
| `TERMINAL_ID`         | `TERMINAL_ID_RISK_NDAY_WINDOW`       | Terminal fraud rate in delayed 1, 7, 30-day windows.          |

A crucial aspect of this project was creating different feature sets optimized for each model architecture. The FFNN baseline model relied entirely on these 15 hand-crafted features, while the CNN model was designed to automatically learn from a more raw data representation.

### 3. Data Augmentation: Addressing Class Imbalance

The severe class imbalance rendered simple training approaches unstable. This necessitated an advanced data augmentation strategy to create a richer, more balanced training set.

#### A. Evaluation of Standard Methods

*   **Random Upsampling/Downsampling:** This was rejected as duplicating samples introduces no new information and leads to overfitting, while removing samples causes significant information loss.

*   **SMOTE & Temporal SMOTE (T-SMOTE):** These methods were also deemed unsuitable for the primary CNN model because their interpolation logic fundamentally disrupts the realistic, sequential nature of fraud events. However, standard **SMOTE was retained for training the baseline FFNN**, which relies on pre-engineered features and is less sensitive to this issue.

#### B. Proposed Solution: Seeded-Sequence Upsampling

To overcome these limitations, a custom augmentation technique was developed to **inject realistic, entire sequences of fraud** into "safe" periods of the timeline.

- **1.  Seed Identification:** Real fraud sequences were extracted from the data, defined as a continuous series of fraudulent transactions for a single customer or terminal.
- **2.  Locating Insertion Points:** The algorithm identifies "safe" 15-day windows with a very low number of existing fraud cases to serve as insertion points.
- **3.  Sequence Injection:** The seed sequence is anchored to the insertion point's timestamp, and all subsequent original transactions are chronologically pushed back.
- **4.  Noise Injection for Robustness:** A small amount of random noise is added to the `TX_AMOUNT` and `TX_DATETIME` of injected transactions to prevent the model from overfitting.
- **5.  Execution Order:** The augmentation is performed sequentially—first for all terminal-based seeds, then for all customer-based seeds. This deliberate ordering, combined with a cooldown mechanism, prevents unrealistic overlaps and allows the model to learn plausible causal relationships.

## Model Development & Architecture
### 1. Overview

The modeling strategy for this project was to first establish strong performance benchmarks with standard models, and then introduce a more sophisticated deep learning architecture designed to overcome their limitations.

#### **A. The Rationale for a Deep Learning Approach**

While tree-based ensembles like XGBoost and Random Forest are industry benchmarks for tabular fraud detection, neural networks offer several unique advantages that are highly relevant for building robust, scalable, and intelligent real-world fraud systems.

*   **1. Representation Learning & End-to-End Training**
    Traditional models are heavily reliant on expert, manual feature engineering to capture historical patterns. In contrast, deep learning models like CNNs excel at **automatic representation learning**—they can learn relevant, hierarchical features directly from raw or semi-raw data.

    > This allows for **end-to-end training**, where the feature extraction and classification components are optimized together. This approach enables the model to discover complex and subtle fraud patterns that manual feature engineering might miss, which was a primary motivator for using a CNN in this project.

*   **2. Incremental Learning**
    Tree-based models are generally "batch" learners, requiring complete retraining on the entire dataset to incorporate new data. This is resource-intensive and can be problematic for data regulation.

    > Neural networks, by their nature, are trained iteratively and can be easily updated on new chunks of data. This **incremental learning** capability is far more efficient for production environments where models must constantly adapt to new fraud tactics.

*   **3. Ensemble Diversity**
    Neural networks and tree-based models learn different types of patterns from data. Even if their overall performance is similar, their predictions often differ on specific cases.

    > This makes neural networks an excellent candidate for **ensembles (or "stacking")**. By combining the predictions of a neural networks, the diversity of the two approaches can lead to a final model that is more accurate and robust than either one alone.

    
#### **B. How models were trained and validated**
- **1**. In real life , a label of a transactions ( fraud or not) is only known after investigation of inspectors due to customer complain, this process on average takes about one week. So normally, when predicting, the model wont have the label of the latest week transactions. To mimic this behaviour and ensure no bias, in training we have to make a 7 day delay after the traning data to start the validation data.
- **2**. In real life, in order to combat Concept Drift, only training data from the a number of closest weeks are used to train and predict the data of the validation week, the number serves as a hyperparameter. Although the Incremental learning capacity of Neural networks models automatically help solving this problem, we should add a mechanism to ensure best practice. However, in our problem there are no Concept Drift, hence we can train on more data to make model learning better.


#### **C. Metrics to evaluate models**
- **1.** Due to the imbalance nature of the dataset, accuracy are useless and not practical to be used to evaluate. This is because even a model that all transactions normal it would still reach 99,99% accuracy. 
- **2.** Hence, special metrics like recall, precision and F1 scores are  utilized, the model must reach a threshold for recall  while not below a threshold for precision to ensure it catches most fraud transactions while not flagging too much normal transactions(lead to displeasing customer).
- **3.** Other metrics used to evaluate overall prediction power of the models are AUCROC, Precision-Recall Curve AUC.
 
#### **D. How to fetch data for the model**
- **1. Fetching Data for Feed-Forward Neural Network:** Use Dataset and DataLoader class of Pytorch to train in batch and get prediction features in torch.tensor format. All the training data are transformed once and saved to be reused for training. the code of this is stored at fraud_detection/src/components/nn_data_ingestion.py in class  FraudDetectionDataset()
- **2. Fetching Data for Bagging CNN-FFNN Neural Network:** :Use Dataset and DataLoader class of Pytorch to train in batch and get prediction features in torch.tensor format. But it is modifed to fetch data in 3 different pipe, customer_history, terminal_history, and current_tx_features. customer_history and terminal_history return a sequence of 0 and 1, label of TX_FRAUD of history. the length of those are tuned by hyperparameter c_len and t_len ( default set to 7).the current_tx_features fetch 15 transformed features like that of the FFNN. The code for this is stored at fraud_detection/src/components/nn_data_ingestion.py in class  BaggingSequentialFraudDetectionDataset()

### 2. Feed-Forward Neural Network

As a strong deep learning baseline, a standard Feed-Forward Neural Network (FFNN) was implemented using PyTorch and PyTorch Lightning. This model's purpose is to evaluate the predictive power of the 15 manually engineered features on their own.

#### **A. Architecture**

The FFNN is a sequential model composed of fully-connected layers designed to process the flat input vector of features. The architecture, defined within the PyTorch Lightning module, is as follows:

*   **Input Layer:** A linear layer that accepts an input vector of size 15 (for the 15 engineered features).
*   **Hidden Layer 1:**
    *   A `Linear` layer transforms the input into a higher-dimensional space (`hidden_size`).
    *   A `ReLU` activation function introduces non-linearity.
    *   `BatchNorm1d` is applied to stabilize learning and improve generalization.
    *   A `Dropout` layer (p=0.4) is used to prevent overfitting by randomly zeroing out a fraction of neurons during training.
*   **Hidden Layer 2:**
    *   A second `Linear` layer further processes the features, reducing the dimensionality by half.
    *   This is followed by another `ReLU` activation and `BatchNorm1d`.
*   **Output Layer:**
    *   A final `Linear` layer projects the features down to a single output neuron.
    *   This output represents the raw "logit" score for the fraud prediction.

#### **B. Training & Optimization**

The model's training process is managed by PyTorch Lightning and incorporates several key components:

*   **Loss Function:** The model is trained using `binary_cross_entropy_with_logits`. This loss function is numerically stable and combines a Sigmoid activation with the binary cross-entropy loss in a single step.
*   **`pos_weight` as hyperparameter:**  A `pos_weight` argument is passed to the loss function. This increases the penalty for misclassifying the fraud class, forcing the model to pay more attention to these crucial examples. Note that we already used SMOTE to train this model, so the ratio is not imbalance anymore, but we still add this in to serve as a hyperparameter to tune to make the model pay more attention to fraud transactions.
*   **Optimizer:** The `Adam` optimizer is used to update the model's weights, with a learning rate of `0.001`.
*   **Metrics & Logging:** During both training and validation, a comprehensive set of metrics is tracked using `torchmetrics` and logged for monitoring:
    *   `AUROC` (Area Under the ROC Curve)
    *   `Precision` & `Recall`
    *   `F1-Score`
    *   The model also logs the average prediction probability (`avg_prob`) and the gradient norms (`grad_...`) to monitor for issues like vanishing or exploding gradients.

#### **C. Results & Analysis**

The model was trained and evaluated using an iterative, rolling-window approach to simulate a real-world production environment. For this analysis, we focus on a representative iteration:

*   **Training Period:** 5 weeks (starting from 2018-05-01)
*   **Testing Period:** The subsequent 1 week

##### **1. Performance with Rolling-Window Training**

The FFNN achieved strong performance with minimal hyperparameter tuning, outperforming standard tree-based baselines like XGBoost and Random Forest on this task.

| AUCROC   | F1       | Precision | Recall   | Top100k_Precision |
| :------- | :------- | :-------- | :------- | :---------------- |
| 0.965105 | 0.449915 | 0.309218  | 0.825545 | 0.509             |

**Key Insights:**

*   **High Recall:** The standout result is a **Recall of 0.825**, indicating that the model successfully identifies over 82% of all fraudulent transactions, which is critical for minimizing financial loss.
*   **Precision and the "Fade-Out" Effect:** The lower Precision of 0.309 is largely attributable to a "fade-out" effect. After a sequence of fraudulent transactions, an entity (`CUSTOMER_ID` or `TERMINAL_ID`) remains "suspicious" for a period, leading to some subsequent normal transactions being flagged as false positives. This is an acceptable trade-off in many production systems, where the cost of missing fraud outweighs the minor inconvenience of a false positive.
*   **Prediction Confidence:** An analysis of the prediction probabilities on the validation set reveals that the model learns a clear separation between the two classes. Most legitimate transactions receive a fraud probability between 0.0 and 0.3, while most fraudulent transactions are confidently assigned a probability close to 1.0.

##### **2. Contrasting Experiment: Training on the Full Dataset**

To validate the rolling-window approach, a second experiment was conducted by training the model on the near-full historical dataset. The performance degraded significantly.

| AUCROC   | F1       | Precision | Recall   | Top100k_Precision |
| :------- | :------- | :-------- | :------- | :---------------- |
| 0.849076 | 0.476309 | 0.398939  | 0.590909 | 1.0               |

**Key Insights:**

*   **Critical Drop in Recall:** While Precision saw a slight increase, the **Recall dropped dramatically to 0.59**. An analysis of the predictions showed a sharp spike in fraud cases being assigned a 0% probability. This indicates that the model, when trained on a massive, static dataset, fails to learn the more nuanced, recent fraud patterns and misses a significant portion of fraudulent activity.

##### **3. Strategic Recommendation**

The iterative, rolling-window training approach is demonstrably superior for this FFNN architecture.

The optimal production strategy would be to implement a **CI/CD pipeline for automated retraining** every few weeks. This approach offers several advantages:
*   **Adaptability:** It effectively combats **concept drift** by keeping the model up-to-date with the latest fraud patterns.
*   **Efficiency:** The training process remains fast and computationally inexpensive.

The primary trade-off is a limited long-term memory; the model may be less effective at recognizing fraud patterns that reappear after a long absence. However, for most dynamic fraud environments, prioritizing recent data is the more effective strategy.
### 3. Bagging CNN-FFNN: A Hybrid Mixture-of-Experts Architecture

The primary model for this project is a novel, hybrid architecture that combines Convolutional Neural Networks (CNNs) for sequence analysis with Feed-Forward Networks (FFNNs) in a **Mixture of Experts (MoE)** framework. The entire model is built as a unified `LightningModule` for robust training.

The core idea is to process three distinct streams of information in parallel—customer history, terminal history, and the current transaction—and then use a "gating network" to intelligently weigh the importance of each stream's output for the final prediction.

#### **A. Architecture**

The model is composed of three parallel "expert" streams that feed into a final weighted decision layer.



**1. Stream 1 & 2: Historical Analysis via Sequence CNNs**

These two identical streams are designed to find fraudulent patterns in the historical transaction sequences of customers and terminals.

*   **Input:** The input is a sequence of historical transaction outcomes (1 for fraud, 0 for non-fraud) for a given customer or terminal.
*   **Embedding Layer:** Each outcome (0 or 1) is first converted into a dense vector representation using an `nn.Embedding` layer. This allows the model to learn a richer representation for "fraud" and "non-fraud" than a simple binary flag.
*   **`SequenceCNN` Module:**
    *   A **1D Convolutional Layer (`Conv1d`)** slides across the sequence of embeddings, acting as a powerful pattern detector. It is designed to learn to recognize specific sequential patterns indicative of fraud (e.g., a sudden burst of fraudulent activity).
    *   An **Adaptive Max Pooling Layer** then summarizes the output of the convolutional layer, identifying the single most "fraud-like" signal detected anywhere in the sequence and creating a fixed-size embedding.
*   **`ExpertFFNN` Head:** The resulting embedding from the CNN is passed to a small, reusable "expert" FFNN, which outputs a single logit score representing the fraud risk based solely on that historical sequence.

**2. Stream 3: Current Transaction Analysis via FFNN**

This stream provides a "context-free" analysis of the current transaction.

*   **Input:** The 15 engineered features of the single, current transaction.
*   **`ExpertFFNN` Head:** The features are passed directly to another `ExpertFFNN`, which outputs a logit score based only on the immediate information.

**3. The Gating Network: Dynamically Weighing the Experts**

This is the most critical component of the architecture. Instead of simply averaging the three expert opinions, the gating network learns to *dynamically trust* each expert based on the input data.

*   **Input:** It takes the rich evidence from all three streams as input: the summarized embeddings from the two CNNs and the raw features of the current transaction.
*   **Logic:** It uses a small neural network to process this combined evidence.
*   **Output:** It outputs three weights (one for each expert). For a given transaction, if the terminal history looks highly suspicious, the gating network might learn to assign a higher weight to the terminal expert's opinion.

**4. Final Prediction**

The final output logit is calculated as a **weighted sum** of the three individual expert logits, using the weights provided by the gating network. This allows the model to be flexible, relying more on historical context for some transactions and more on the current transaction's details for others.

#### **B. Training and Optimization**

The training and evaluation logic is cleanly encapsulated within the `LightningModule`.

*   **Loss Function:** The model is optimized using `binary_cross_entropy_with_logits`, which is numerically stable and ideal for binary classification. The crucial `pos_weight` parameter is used to heavily penalize misclassifications of the rare fraud class.
*   **Optimizer:** The `Adam` optimizer is used to manage the learning process.
*   **Comprehensive Logging:** The `training_step` and `validation_step` methods are configured to log key performance metrics like `recall`, `precision`, and `auroc` on each epoch, providing a clear view of the model's learning progress via TensorBoard.

#### **C. Results & Analysis**

The Bagging CNN-FFNN model was evaluated using the same iterative, rolling-window approach to provide a direct comparison with the FFNN baseline.

##### **1. Performance with Rolling-Window Training**

When trained on the same small, iterative data window (5 weeks of training, 1 week of testing), the complex hybrid model struggled to deliver competitive performance.

| AUCROC   | F1       | Precision | Recall   | Top100k_Precision |
| :------- | :------- | :-------- | :------- | :---------------- |
| 0.918697 | 0.194716 | 0.111275  | 0.778434 | 0.271             |

**Key Insights:**

*   **Underperformance Compared to FFNN:** The results are notably worse than the simpler FFNN. This is a classic example of a complex, "data-hungry" model **underfitting** when provided with insufficient data to learn its numerous parameters effectively.
*   **Prediction Distribution:** While the model correctly assigns low fraud probabilities (<0.3) to most normal transactions, the probabilities for fraudulent transactions are more fluctuated and less confident than the FFNN's. This is an expected result of the Mixture-of-Experts architecture, as it merges potentially conflicting "opinions" from its three internal experts.

##### **2. Performance with Full Dataset Training**

Recognizing the model's need for more data, a second experiment was conducted by training it on the near-full historical dataset. The results improved dramatically, confirming the initial hypothesis.

| AUCROC   | F1       | Precision | Recall   | Top100k_Precision |
| :------- | :------- | :-------- | :------- | :---------------- |
| 0.964226 | 0.655679 | 0.589549  | 0.738518 | 0.841             |

**Key Insights:**

*   **Superior Overall Performance:** With a sufficient volume of data, the hybrid model's performance surpasses the FFNN baseline across most key metrics, especially **F1-Score (0.65 vs 0.45)** and **Precision (0.59 vs 0.31)**.
*   **Reduced "Missed" Frauds:** The most salient improvement is in the distribution of fraud prediction probabilities. The spike at 0% probability for true fraud cases is **significantly smaller** than that of the FFNN. This indicates that the CNN-FFNN is much less likely to completely miss a fraudulent transaction, making it a more reliable detection system.

##### **3. Conclusion & Strategic Recommendation**

The two experiments reveal a critical trade-off:
*   The **FFNN** is highly effective with small, recent data but fails to generalize on a large, static dataset.
*   The **Bagging CNN-FFNN** is a more powerful and reliable model but requires a large volume of data to learn effectively.

The optimal production strategy for this advanced model would be an **automated, incremental training pipeline**.
*   **Recommendation:** Instead of retraining from scratch on a small rolling window, the model should be continuously updated by training on newly arriving data (e.g., daily or weekly).
*   **Benefits:** This approach allows the model to **accumulate knowledge** from the large historical dataset while also **integrating new information** to adapt to concept drift. It combines the best of both worlds: robustness from historical data and adaptability to new fraud patterns.