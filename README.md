# TrustFed
 Privacy-Preserving Federated Learning Healthcare System
# TrustFed: Privacy-Preserving Federated Learning with XAI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Federated%20Learning-green)
![Security](https://img.shields.io/badge/Security-Homomorphic%20Encryption-red)
![XAI](https://img.shields.io/badge/XAI-LIME-orange)

## 🏥 Overview

**TrustFed** is a decentralized, privacy-preserving healthcare AI framework designed to tackle the dual challenges of data privacy and model transparency. 

It implements a simulation of **Federated Learning (FL)** combined with **Lightweight Additive Homomorphic Encryption (HE)** to securely train a global heart disease prediction model across multiple hospitals without ever sharing raw patient data. To ensure clinical trust, the system integrates **Explainable AI (XAI)** at the inference stage, allowing doctors to interpret "black box" model decisions.

This project is an implementation and extension of the research paper *"FedHealthcare: Federated Learning and Lightweight Additive Homomorphic Encryption"* (Bhattacharjee et al., 2025).

## 🚀 Key Features

* **Federated Learning Simulation:** Simulates a network of 3 independent hospital clients training locally on private data shards.
* **Custom Cryptography:** Implements **Lightweight Additive Homomorphic Encryption** ($C = W + K$) from scratch using NumPy to secure model gradients during aggregation.
* **Explainable AI (XAI):** Integrates **LIME (Local Interpretable Model-agnostic Explanations)** to provide client-side verification of specific predictions.
* **High Performance:** Achieves **>90% accuracy** on a synthetic Heart Disease dataset with minimal encryption overhead (~5%).
* **Data Visualization:** Includes Correlation Matrices for feature analysis and Confusion Matrices for performance evaluation.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Logistic Regression), NumPy, Pandas
* **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

## ⚙️ Architecture Workflow

1.  **Local Training:** Each client trains a local Logistic Regression model on their private dataset.
2.  **Encryption:** Clients encrypt their model weights by adding a secure random noise mask ($W_{encrypted} = W_{raw} + K_{mask}$).
3.  **Secure Aggregation:** The central server averages the encrypted weights from all clients. Due to the additive property of the encryption, the server aggregates knowledge without seeing individual model parameters.
4.  **Global Update:** The aggregated global model is decrypted (by subtracting the aggregated masks) and sent back to clients.
5.  **Client-Side Verification:** The client uses **LIME** to generate feature importance plots, verifying that the model's diagnosis is based on valid medical factors (e.g., Cholesterol, Blood Pressure) rather than noise.

## 📊 Results

* **Model Accuracy:** 92.5% (simulated on Test Set)
* **Privacy Preserved:** Raw patient data never leaves the local client.
* **Interpretability:** Successfully generated LIME plots identifying top risk factors for individual patients.

*(You can add screenshots of your LIME plots or Confusion Matrix here)*

## 📦 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Abhishek0489/TrustFed.git](https://github.com/Abhishek0489/TrustFed.git)
    cd TrustFed
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn lime matplotlib seaborn
    ```

3.  **Run the Simulation**
    * Open `TrustFed_Simulation.ipynb` in Jupyter Notebook or Google Colab.
    * Ensure `synthetic_heart_disease_dataset.csv` is in the same directory (or upload it to Colab).
    * Run all cells to execute the FL training loop and generate XAI explanations.

## 📄 References

This project is based on the methodology proposed in:
> [cite_start]*FedHealthcare: Federated Learning and Lightweight Additive Homomorphic Encryption-Based Privacy-Preserving Healthcare Framework* (Security and Privacy, Wiley, 2025). [cite: 5, 27]

## 👤 Author

**Abhishek Oraon**
* [LinkedIn](https://linkedin.com/in/abhishek-oraon-bba42b272)
* [GitHub](https://github.com/Abhishek0489)

---
*This project was developed as a Minor Project for B.Tech CSE.*
