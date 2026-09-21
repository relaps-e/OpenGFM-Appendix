# OpenGFM: A Unified Evaluation Framework for Graph Foundation Models

This repository accompanies **Rethinking Graph Foundation Models: A Pipeline-Level Taxonomy and Capability-Oriented Benchmark**. It collects the method overview, environment, hyperparameter settings, prompt designs, and numerical results used by OpenGFM.

## 🛠️ Environment Setup & Requirements

To reproduce the experiments, please set up the environment using the following dependencies.

```bash
torch==2.0.1+cu118
torch_geometric==2.5.3
pyg-lib==0.4.0+pt20cu118
matplotlib==3.8.4
networkx==3.3
numba==0.60.0
numpy==1.26.4
ogb==1.3.6
pandas==1.5.3
scikit-learn==1.5.2
scipy==1.13.1
```

---

## 📊 Comprehensive Comparison of Graph Foundation Models

The following table summarizes the supported tasks, cross-dataset capabilities, and pre-training strategies of the evaluated GFMs.

| Family | Methods | Node Cls. | Link Pred. | Graph Cls. | Cross-Dataset Capability | Pre-training Strategy | Adaptation |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **LLM-Enhanced GNN** | TAPE | ✅ | ❌ | ❌ | ❌ | Supervised Pretrain. | Finetune |
| | ENGINE | ✅ | ❌ | ❌ | ❌ | Supervised Pretrain. | Finetune |
| | OFA | ✅ | ✅ | ✅ | ✅ (Multi-Source) | Supervised Pretrain. | Graph Prompting |
| **Graph-Enhanced LLM** | LLaGA | ✅ | ✅ | ❌ | ✅ (Multi-Source) | Pre-trained LLM | In-context |
| | GraphGPT | ✅ | ✅ | ❌ | ✅ (Multi-Source) | Pre-trained LLM | In-context & Finetune |
| | GraphText | ✅ | ❌ | ❌ | ❌ | Pre-trained LLM | In-context |
| **Graph Prompt Learning** | GPF | ❌ | ❌ | ✅ | ❌ | Graph SSL | Graph Prompting |
| | GraphPrompt | ✅ | ✅ | ✅ | ❌ | Graph SSL | Graph Prompting |
| **Unified GM & Graph MoE**| AnyGraph | ✅ | ✅ | ❌ | ✅ (Multi-Source) | Graph SSL | Finetune |
| | OpenGraph | ✅ | ✅ | ❌ | ✅ (Multi-Source) | Supervised Pretrain. | Finetune |
| | GraphAny | ✅ | ❌ | ❌ | ✅ (Single-Source) | Supervised Pretrain. | N/A |

---

## ⚙️ Hyperparameter Settings

This section details the hyperparameter search spaces and specific dataset configurations used for the evaluated methods. For hyperparameter optimization, we leveraged the Optuna framework to perform an efficient search, using the TPE sampler combined with the ASHA pruner, unless otherwise specified.

### 1. Traditional GNN Baselines
*(Epochs are set to 2000 for arXiv, and early stopping is applied across datasets. Weight Decay (WD) is omitted/defaulted for arXiv).*

**GCN Hyperparameters**

| Dataset | FeatureTrans. | ResNet | Norm. | Dropout | Layers | Hid. | LR. | WD. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cora-PyG** | False | False | False | 0.7 | 3 | 512 | 0.001 | 5e-4 |
| **Cora-TAG** | False | False | False | 0.4 | 2 | 128 | 0.005 | 0 |
| **Citeseer-PyG** | False | False | False | 0.5 | 2 | 512 | 0.001 | 5e-4 |
| **Citeseer-TAG** | False | False | False | 0.3 | 2 | 512 | 0.01 | 5e-4 |
| **Pubmed-PyG** | False | False | False | 0.7 | 2 | 256 | 0.005 | 5e-4 |
| **Pubmed-TAG** | False | False | False | 0.1 | 2 | 512 | 0.005 | 0 |
| **chameleon** | 1(Linear+Relu) | False | False | 0.3 | 5 | 512 | 0.005 | 5e-4 |
| **squirrel** | 1(Linear+LeakyRelu) | True | True | 0.7 | 4 | 256 | 0.005 | 0 |
| **arXiv** | False | True | True | 0.5 | 5 | 512 | 0.0005 | - |

**GAT Hyperparameters**

| Dataset | FeatureTrans. | ResNet | Norm. | Dropout | Layers | Hid. | LR. | WD. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cora-PyG** | False | False | False | 0.2 | 3 | 512 | 0.001 | 5e-4 |
| **Cora-TAG** | False | False | False | 0.4 | 2 | 256 | 0.001 | 5e-4 |
| **Citeseer-PyG** | False | False | False | 0.5 | 3 | 256 | 0.001 | 5e-4 |
| **Citeseer-TAG** | False | False | False | 0.1 | 2 | 512 | 0.01 | 0 |
| **Pubmed-PyG** | False | False | False | 0.5 | 2 | 512 | 0.01 | 5e-4 |
| **Pubmed-TAG** | False | False | False | 0.5 | 2 | 512 | 0.01 | 5e-5 |
| **chameleon** | 1(Linear+Relu) | False | False | 0.6 | 3 | 512 | 0.01 | 5e-4 |
| **squirrel** | 2(Linear+Sigmoid) | True | True | 0.4 | 7 | 512 | 0.005 | 5e-5 |
| **arXiv** | False | True | True | 0.5 | 4 | 256 | 0.0005 | - |

**GraphSAGE (SAGE) Hyperparameters**

| Dataset | FeatureTrans. | ResNet | Norm. | Dropout | Layers | Hid. | LR. | WD. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cora-PyG** | False | False | False | 0.7 | 3 | 256 | 0.001 | 5e-4 |
| **Cora-TAG** | False | False | False | 0.2 | 4 | 256 | 0.005 | 5e-5 |
| **Citeseer-PyG** | False | False | False | 0.2 | 3 | 512 | 0.001 | 5e-4 |
| **Citeseer-TAG** | False | False | False | 0.3 | 2 | 256 | 0.01 | 5e-5 |
| **Pubmed-PyG** | False | False | False | 0.7 | 4 | 512 | 0.005 | 5e-4 |
| **Pubmed-TAG** | False | False | False | 0.3 | 4 | 512 | 0.001 | 5e-4 |
| **chameleon** | 1(Linear+Relu) | False | False | 0.8 | 3 | 512 | 0.01 | 5e-4 |
| **squirrel** | 1(Linear+LeakyRelu) | True | True | 0.8 | 3 | 256 | 0.01 | 5e-4 |
| **arXiv** | False | True | True | 0.5 | 5 | 256 | 0.0005 | - |

**GraphGPS Hyperparameters**

| Dataset | Local GNN | PE | Layers | Heads | Hidden | Dropout | Attn Dropout | LR | WD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cora-PyG** | GCN | LapPE | 2 | 2 | 64 | 0.2 | 0.1 | 0.001 | 5e-4 |
| **Cora-TAG** | GCN | LapPE | 3 | 4 | 64 | 0.3 | 0.2 | 0.005 | 0 |
| **Citeseer-PyG** | GCN | LapPE | 3 | 4 | 64 | 0.4 | 0.3 | 0.001 | 5e-4 |
| **Citeseer-TAG** | GCN | LapPE | 4 | 4 | 64 | 0.1 | 0.4 | 0.01 | 5e-4 |
| **Pubmed-PyG** | GCN | LapPE | 3 | 4 | 64 | 0.5 | 0.2 | 0.005 | 5e-4 |
| **Pubmed-TAG** | GCN | LapPE | 3 | 4 | 64 | 0.2 | 0.5 | 0.005 | 0 |
| **chameleon** | GCN | LapPE | 3 | 4 | 96 | 0.2 | 0.5 | 0.0005 | 1e-5 |
| **squirrel** | GCN | LapPE | 2 | 4 | 64 | 0.2 | 0.0 | 0.0005 | 1e-5 |
| **arXiv** | GCN | LapPE | 2 | 4 | 64 | 0.3 | 0.1 | 0.0005 | - |

**GCA Hyperparameters**

| Dataset | FeatureTrans. | ResNet | Norm. | Dropout | Layers | Hid. | LR. | WD. | $\mu_1$ | $\mu_2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cora-PyG** | False | False | False | 0.7 | 3 | 512 | 0.001 | 5e-4 | 0.1 | 0.2 |
| **Cora-TAG** | False | False | False | 0.4 | 2 | 128 | 0.005 | 0 | 0.1 | 0.2 |
| **Citeseer-PyG** | False | False | False | 0.5 | 2 | 512 | 0.001 | 5e-4 | 0.05 | 0.1 |
| **Citeseer-TAG** | False | False | False | 0.3 | 2 | 512 | 0.01 | 5e-4 | 0.05 | 0.1 |
| **Pubmed-PyG** | False | False | False | 0.7 | 2 | 256 | 0.005 | 5e-4 | 0.1 | 0.2 |
| **Pubmed-TAG** | False | False | False | 0.1 | 2 | 512 | 0.005 | 0 | 0.1 | 0.2 |
| **chameleon** | 1(Linear+Relu) | False | False | 0.3 | 5 | 512 | 0.005 | 5e-4 | 0.2 | 0.4 |
| **squirrel** | 1(Linear+LeakyRelu) | True | True | 0.7 | 4 | 256 | 0.005 | 0 | 0.2 | 0.4 |
| **arXiv** | False | True | True | 0.5 | 5 | 512 | 0.0005 | - | 0.2 | 0.4 |

### 2. LLM-Enhanced GNNs
* **TAPE:** We searched over number of layers in `{2, 3, 4}`, hidden dimension in `{64, 128, 256}`, learning rate in `{1e-2, 2e-4, 1e-4}`, dropout rate in `{0.3, 0.5, 0.6}`, and weight decay in `{0, 5e-4}`. Models were trained for 1000 epochs with an early stopping patience of 50.
* **ENGINE:** We searched over number of layers in `{1, 2, 3}`, hidden dimension in `{64, 128}`, learning rate fixed at `5e-5`, dropout rate in `{0.3, 0.5, 0.6}`, and weight decay fixed at `5e-4`. Models were trained for 200 epochs.
* **OFA:** We searched over number of layers in `{6, 7}`, hidden dimension in `{512, 768}`, learning rate in `{1e-3, 1e-4, 5e-5}`, and dropout rate in `{0.1, 0.15, 0.3}`. Models were trained for 100 epochs.

### 3. Graph-Enhanced LLMs
* **GraphText:** We followed the original paper's settings: learning rate = `5e-5`, dropout = `0.5`, subgraph size = `3`, and max sequence length = `1024`.
* **GraphGPT:** We followed the original paper's settings: learning rate = `2e-3`, warmup ratio = `0.03`, and number of training epochs = `3`.
* **LLaGA:** We followed the original paper's settings: learning rate = `2e-3`, warmup ratio = `0.03`, number of hops = `2`, sample neighbor size = `10`, batch size = `16`, learning rate scheduler = `cosine`, and max length = `4096`.

### 4. Graph Prompt Learning
* **GPF & GPrompt:** We fixed the number of layers to 2 and hidden dimension to 128. We searched over learning rate selected from `10^uniform(-3, -1)`, weight decay selected from `10^uniform(-5, -6)`, and batch size selected from `{32, 64, 128}`.

### 5. Unified Graph Model & Graph Mixture-of-Models
* **OpenGraph:** For standard inference, we directly used the pre-trained models `pretrn_gen0`, `pretrn_gen1`, and `pretrn_gen2`. For OpenGraph-FT, we ensured model parameter updates by adding classification heads and conducting full-parameter fine-tuning on the test datasets. We searched over learning rate in `{1e-2, 1e-4, 2e-5, 1e-5}`, batch size in `{32, 64, 128, 256, 512, 1024}`, number of training epochs in `{50, 250, 500, 1000}`, and dropout rate in `{0.1, 0.3, 0.5, 0.7}`.
* **AnyGraph:** For standard inference, we directly used the pre-trained models `pretrain_link1` and `pretrain_link2`. For AnyGraph-FT, we conducted full-parameter fine-tuning on the test datasets. We searched over learning rate in `{1e-2, 1e-4, 2e-5, 1e-5}`, batch size in `{32, 64, 128, 256, 512, 1024}`, number of training epochs in `{50, 250, 500, 1000}`, and dropout rate in `{0.1, 0.3, 0.5, 0.7}`.
* **GraphAny:** For standard inference, we used the pre-trained models `graph_any_arxiv.pt`, `graph_any_cora.pt`, `graph_any_product.pt`, and `graph_any_wisconsin.pt`. For GraphAny-FT, we searched over training steps in `{250, 500, 750, 1000}`, hidden dimension in `{32, 64, 128, 256, 512}`, number of MLP layers in `{1, 2}`, entropy normalization parameter in `{1, 2}`, and number of per-label examples in `{3, 5}`. The attention temperature was fixed at `5`.

---

## 📝 LLM-Based Methods Prompt Designs

Below are the exact prompt templates used for Text-Attributed Graph (TAG) datasets across different LLM-based methodologies.

### TAPE
* **Cora-TAG:** `Question: Which of the following sub-categories of AI does this paper belong to: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text. Answer:`
* **Citeseer-TAG:** `Question: Which of the following sub-categories of CS does this paper belong to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence)? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text. Answer:`
* **PubMed-TAG:** `Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option. Answer:`

### GraphGPT
* **Cora-TAG:** `Given a citation graph: <graph>, where the 0-th node is the target paper, with the following information: <raw text>. Question: Which of the following specific research areas does this paper belong to: <labels>. Directly give the full name of the most likely category of this paper.`
* **Citeseer-TAG:** `Given a citation graph: <graph>, where the 0-th node is the target paper, with the following information: <raw text>. Question: Which of the following computer science research fields does this paper belong to: <labels>. Directly give the full name of the most likely field of this paper.`
* **PubMed-TAG:** `Given a biomedical citation graph: <graph>, where the 0-th node is the target paper, with the following information: <raw text>. Question: Which of the following medical research categories does this paper belong to: <labels>. Directly give the full name of the most likely category of this paper.`

### LLaGA
* **Cora-TAG:** `Given a node-centered citation graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: <labels>. Please tell me which research area the center node belongs to?`
* **Citeseer-TAG:** `Given a node-centered citation graph: <graph>, each node represents a computer science paper, we need to classify the center node into 6 classes: <labels>. Please tell me which research field the center node belongs to?`
* **PubMed-TAG:** `Given a node-centered biomedical graph: <graph>, each node represents a medical paper, we need to classify the center node into 3 classes: <labels>. Please tell me which medical category the center node belongs to?`

### GraphText
* **Cora-TAG:** `You are a helpful assistant that classifies the topic of an academic paper based on the labels of the cited papers. You are going to choose the correct answer from several choices: [A: Rule_Learning, B: Neural_Networks, C: Case_Based, D: Genetic_Algorithms, E: Theory, F: Reinforcement_Learning, G: Probabilistic_Methods]. Here are a few examples: <information><feature><center node>['A']</center node><1st feature similarity graph>['A', 'A', 'A']</1st feature similarity graph></feature></information><question>What's the topic of this paper?</question><answer>A</answer> Remaining examples... Now let's answer the question below: <information><feature><center node>['C']</center node><1st feature similarity graph>['C', 'B', 'B']</1st feature similarity graph></feature></information> What's the topic of the paper? Valid choices are [A-G]. Remember, your answer should be in the form of <answer>C</answer>.`
* **Citeseer-TAG:** `You are a helpful assistant that classifies the topic of an academic paper based on the labels of the cited papers. You are going to choose the correct answer from several choices: [A: Agents, B: Artificial Intelligence, C: Database, D: Information Retrieval, E: Machine Learning, F: Human Computer Interaction]. Here are a few examples: <information><third-order pseudo labels><center node>['A']</center node><1st feature similarity graph>['A', 'A', 'A']</1st feature similarity graph><ppr>['A', 'B', 'A']</ppr></third-order pseudo labels></information><question>What's the topic of academic paper given the information above?</question><answer>A</answer> Remaining examples... Now let's answer the question below: <information><third-order pseudo labels><1st feature similarity graph>['C', 'B', 'B']</1st feature similarity graph><ppr>['C']</ppr></third-order pseudo labels></information> What's the topic of the paper? Valid choices are [A-F]. Remember, your answer should be in the form of <answer>C</answer>.`
* **PubMed-TAG:** `You are a helpful assistant that classifies the topic of a biomedical paper based on the labels of the related papers. You are going to choose the correct answer from several choices of medical categories: [A: Experimental, B: Clinical, C: Other]. Here are a few examples: <information><third-order pseudo labels><center node>['A']</center node><1st feature similarity graph>['A', 'A', 'A']</1st feature similarity graph><ppr>['A', 'B', 'A']</ppr></third-order pseudo labels></information><question>What's the medical category of this paper?</question><answer>A</answer> Remaining examples... Now let's answer the question below: <information><third-order pseudo labels><1st feature similarity graph>['C', 'B', 'B']</1st feature similarity graph><ppr>['C']</ppr></third-order pseudo labels></information> What's the medical category of the paper? Valid choices are [A-C]. Remember, your answer should be in the form of <answer>C</answer>.`

---

## Standard performance results

The following table condenses the node-classification accuracy results by reporting the strongest evaluated GFM candidate and the strongest conventional GNN on each dataset.

| Dataset | Best GFM candidate | Accuracy (%) | Best GNN | Accuracy (%) |
| :--- | :--- | ---: | :--- | ---: |
| Cora-NT | GPF-FT | 83.41 ± 0.25 | GCA | 85.44 ± 0.91 |
| Cora-TAG | ENGINE | 85.17 ± 1.02 | GCA | 85.62 ± 0.83 |
| Citeseer-NT | GIT | 81.97 ± 0.80 | GCA | 75.69 ± 0.83 |
| Citeseer-TAG | ENGINE | 74.55 ± 1.62 | GCA | 77.33 ± 1.11 |
| PubMed-NT | GPF-FT | 77.97 ± 0.49 | GCA | 81.07 ± 0.79 |
| PubMed-TAG | TAPE | 82.35 ± 1.01 | SAGE | 82.30 ± 0.22 |
| arXiv | TAPE | 75.08 ± 0.21 | GCA | 75.88 ± 0.15 |
| Chameleon | GIT | 39.28 ± 2.63 | GAT | 47.18 ± 3.69 |
| Squirrel | GPrompt-FT | 40.11 ± 1.49 | GCA | 45.63 ± 2.30 |

Link prediction reports AP, and graph classification reports accuracy.

| Family | Method | Cora LP | Citeseer LP | PubMed LP |
| :--- | :--- | ---: | ---: | ---: |
| LLM-Enhanced GNN | OFA | 91.84 ± 0.51 | 92.17 ± 0.52 | 94.38 ± 0.21 |
| Graph-Enhanced LLM | LLaGA | 91.37 ± 0.40 | 90.92 ± 0.41 | 93.33 ± 0.15 |
| Graph Prompt Learning | GPrompt | 92.75 ± 0.55 | 91.79 ± 0.38 | 95.33 ± 0.43 |
| Unified Graph Model | GIT | 87.02 ± 0.42 | 87.35 ± 0.38 | 87.23 ± 0.27 |
| Unified Graph Model | OpenGraph | 72.23 ± 0.36 | 76.78 ± 0.33 | 77.35 ± 0.29 |
| Graph Mixture-of-Experts | AnyGraph | 71.79 ± 0.31 | 77.59 ± 0.31 | 74.59 ± 0.20 |
| GNN | GCN | 94.14 ± 0.44 | 95.39 ± 0.62 | 98.42 ± 0.06 |
| GNN | GAT | 94.17 ± 0.45 | 95.85 ± 0.63 | 97.33 ± 0.16 |
| GNN | SAGE | 94.78 ± 0.32 | 97.17 ± 0.26 | 98.80 ± 0.07 |

| Family | Method | IMDB-B | PROTEINS | MUTAG |
| :--- | :--- | ---: | ---: | ---: |
| LLM-Enhanced GNN | OFA | 68.72 ± 3.97 | 72.79 ± 3.97 | 78.72 ± 9.12 |
| Graph Prompt Learning | GPF | 64.50 ± 3.87 | 70.53 ± 2.88 | 72.14 ± 6.14 |
| Graph Prompt Learning | GPrompt | 65.32 ± 5.32 | 68.71 ± 3.81 | 72.19 ± 6.55 |
| Unified Graph Model | GIT | 71.18 ± 4.24 | 72.35 ± 3.51 | 79.51 ± 8.24 |
| Unified Graph Model | GFT | 66.54 ± 4.13 | 69.22 ± 3.27 | 75.87 ± 7.27 |
| GNN | GCN | 67.20 ± 4.80 | 72.86 ± 4.18 | 82.08 ± 12.17 |
| GNN | GAT | 66.40 ± 3.30 | 72.50 ± 2.55 | 79.33 ± 7.89 |
| GNN | SAGE | 68.40 ± 4.10 | 72.68 ± 3.81 | 79.24 ± 6.86 |
| GNN | GCA | 73.20 ± 3.70 | 74.10 ± 3.65 | 88.17 ± 5.45 |

---

## Three-seed TAG transfer results

The transfer comparison uses Cora-TAG, Citeseer-TAG, and PubMed-TAG. Source models use 20 labels per class, while target adaptation uses 1, 5, 10, or 20 labels per class. Every source-target direction is evaluated with model seeds 123, 456, and 789. Each cell reports the mean accuracy gain over matched target-only training, followed by the sample standard deviation across the three seed-level means and the number of positive source-target-seed comparisons out of 18.

### Frozen backbone

| Method | K=1 | K=5 | K=10 | K=20 |
| :--- | ---: | ---: | ---: | ---: |
| GCN | -17.67 ± 1.50 (0/18) | -15.06 ± 0.82 (0/18) | -12.02 ± 0.94 (0/18) | -10.74 ± 0.43 (0/18) |
| GAT | -16.77 ± 0.93 (0/18) | -12.36 ± 1.59 (0/18) | -10.86 ± 0.56 (0/18) | -7.89 ± 1.51 (0/18) |
| ENGINE | -15.79 ± 4.04 (0/18) | -16.74 ± 6.43 (0/18) | -13.46 ± 6.09 (0/18) | -14.57 ± 6.13 (0/18) |
| OFA | +3.07 ± 2.89 (11/18) | +6.66 ± 4.56 (14/18) | +4.27 ± 1.73 (12/18) | -10.40 ± 22.78 (7/18) |
| GPF-Plus | -20.13 ± 0.55 (0/18) | -18.93 ± 2.54 (0/18) | -21.72 ± 2.43 (0/18) | -23.38 ± 0.54 (0/18) |
| GPrompt | -9.09 ± 3.04 (1/18) | -8.11 ± 0.32 (1/18) | -8.98 ± 2.92 (2/18) | -20.25 ± 3.01 (0/18) |
| OpenGraph | -5.49 ± 1.98 (2/18) | -10.64 ± 2.62 (0/18) | -12.92 ± 4.88 (1/18) | -15.35 ± 4.02 (1/18) |
| RiemannGFM | -1.08 ± 2.58 (7/18) | -8.27 ± 2.41 (1/18) | -6.19 ± 1.23 (0/18) | -5.14 ± 1.90 (2/18) |
| GIT | -18.37 ± 0.93 (0/18) | -12.51 ± 2.56 (0/18) | -9.63 ± 2.25 (0/18) | -10.15 ± 1.06 (0/18) |
| SAMGPT | -14.76 ± 0.66 (0/18) | -14.42 ± 0.61 (0/18) | -13.81 ± 1.11 (0/18) | -15.81 ± 2.27 (0/18) |
| GraphAny | -19.60 ± 4.54 (0/18) | -19.66 ± 3.95 (1/18) | -13.10 ± 12.69 (4/18) | -20.82 ± 11.59 (2/18) |
| AnyGraph | -7.02 ± 2.60 (2/18) | -15.26 ± 1.30 (0/18) | -12.61 ± 3.55 (2/18) | -20.63 ± 1.83 (0/18) |

### Full adaptation

| Method | K=1 | K=5 | K=10 | K=20 |
| :--- | ---: | ---: | ---: | ---: |
| GCN | -3.00 ± 0.58 (3/18) | -2.44 ± 0.78 (2/18) | -2.29 ± 1.44 (3/18) | -1.02 ± 0.44 (3/18) |
| GAT | -2.95 ± 1.28 (4/18) | -2.28 ± 1.38 (4/18) | -2.39 ± 0.81 (3/18) | -1.12 ± 0.46 (5/18) |
| ENGINE | -1.78 ± 1.50 (7/18) | -1.63 ± 0.80 (6/18) | -1.35 ± 0.29 (6/18) | -0.22 ± 0.59 (9/18) |
| OFA | +5.63 ± 1.70 (12/18) | +9.79 ± 1.55 (15/18) | +17.36 ± 5.25 (15/18) | +10.67 ± 29.46 (13/18) |
| GPF-Plus | -2.84 ± 0.34 (6/18) | -2.53 ± 2.05 (5/18) | -1.19 ± 0.65 (5/18) | -0.80 ± 0.78 (6/18) |
| GPrompt | -0.77 ± 1.91 (7/18) | -0.55 ± 0.77 (7/18) | -0.53 ± 0.71 (4/18) | -0.47 ± 1.25 (8/18) |
| OpenGraph | -0.94 ± 1.58 (6/18) | +0.21 ± 0.82 (9/18) | +1.90 ± 2.16 (14/18) | +0.33 ± 1.80 (11/18) |
| RiemannGFM | -1.55 ± 0.77 (7/18) | +0.18 ± 0.41 (9/18) | -0.77 ± 1.05 (6/18) | +0.69 ± 1.15 (8/18) |
| GIT | -4.57 ± 0.69 (4/18) | -0.80 ± 1.20 (6/18) | +1.00 ± 2.10 (13/18) | -0.77 ± 1.15 (6/18) |
| SAMGPT | -1.49 ± 0.59 (4/18) | -1.96 ± 0.51 (6/18) | -1.06 ± 0.22 (6/18) | -1.08 ± 0.65 (4/18) |
| GraphAny | -0.72 ± 0.45 (5/18) | -1.24 ± 1.40 (2/18) | -1.81 ± 0.64 (0/18) | -1.01 ± 0.46 (2/18) |
| AnyGraph | -2.88 ± 4.16 (8/18) | -5.50 ± 4.95 (3/18) | +0.09 ± 0.87 (8/18) | -1.63 ± 1.71 (7/18) |

### Native prompt update

| Method | K=1 | K=5 | K=10 | K=20 |
| :--- | ---: | ---: | ---: | ---: |
| GPF-Plus | -10.53 ± 1.74 (0/18) | -10.96 ± 1.47 (1/18) | -8.14 ± 0.39 (0/18) | -4.39 ± 1.07 (1/18) |
| GPrompt | -7.93 ± 2.64 (0/18) | -7.40 ± 1.04 (2/18) | -5.37 ± 2.45 (3/18) | -7.74 ± 3.74 (0/18) |
| SAMGPT | -7.01 ± 0.36 (1/18) | -7.24 ± 0.91 (0/18) | -5.77 ± 0.71 (0/18) | -6.91 ± 1.01 (0/18) |

Frozen reuse remains below matched target-only training for most methods. Full adaptation closes much of this gap and yields positive mean transfer for OFA across all four label budgets and for selected budgets of OpenGraph, RiemannGFM, GIT, and AnyGraph. OFA at K=20 also has the largest cross-seed variation, so its positive mean should be read together with the reported standard deviation.

---

## Efficiency results

Efficiency is measured on Cora-TAG, Citeseer-TAG, and PubMed-TAG using an NVIDIA A100-PCIE-40GB. Target adaptation uses K=10. The tables separate preparation, target fitting, training-phase computation, full-test-graph inference, throughput, peak process-level GPU memory, parameter updates, storage, and repeated use. Entries with “±” report the mean and sample standard deviation across the three TAG datasets.

### Target fitting time by update mode

| Method | Scratch fit (s) | Frozen fit (s) | Full fit (s) | Prompt/adapter fit (s) |
| :--- | ---: | ---: | ---: | ---: |
| GCN | 3.59 ± 1.01 | 7.79 ± 2.02 | 4.18 ± 2.32 | N/A |
| GAT | 3.25 ± 0.07 | 11.00 ± 2.78 | 5.92 ± 1.10 | N/A |
| ENGINE | 4.81 ± 1.48 | 11.17 ± 6.99 | 5.40 ± 2.11 | N/A |
| OFA | 30.83 ± 21.87 | 9.81 ± 7.03 | 11.25 ± 6.27 | N/A |
| GraphGPT | 164.79 ± 42.70 | 179.88 ± 7.06 | N/A | 130.75 ± 20.89 |
| LLaGA | 87.33 ± 43.60 | 51.71 ± 2.81 | N/A | 65.69 ± 10.94 |
| GPF-Plus | 3.33 ± 1.05 | 6.14 ± 4.73 | 3.35 ± 0.48 | 3.91 ± 0.39 |
| GPrompt | 2.82 ± 0.31 | 3.61 ± 0.81 | 2.70 ± 0.60 | 5.14 ± 1.23 |
| OpenGraph | 3.79 ± 1.84 | 4.40 ± 2.02 | 2.52 ± 1.24 | N/A |
| RiemannGFM | 16.01 ± 2.81 | 18.95 ± 1.10 | 15.63 ± 2.98 | N/A |
| GIT | 2.75 ± 1.27 | 2.58 ± 0.24 | 1.51 ± 0.12 | N/A |
| SAMGPT | 3.58 ± 0.65 | 3.03 ± 0.52 | 3.97 ± 0.85 | 5.56 ± 0.98 |
| GraphAny | 23.55 ± 4.02 | 5.95 ± 0.05 | 22.28 ± 3.32 | N/A |
| AnyGraph | 57.96 ± 51.60 | 5.95 ± 3.92 | 56.84 ± 50.46 | N/A |

### Native target-update outcomes and phase times

| Method | Update | Accuracy (%) | Macro-F1 (%) | Balanced Acc. (%) | Preparation (s) | Target fit (s) | Train phase (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | Full | 72.27 ± 5.50 | 69.91 ± 7.50 | 71.75 ± 9.27 | 1.79 ± 0.30 | 4.18 ± 2.32 | 1.85 ± 1.09 |
| GAT | Full | 72.27 ± 6.11 | 70.62 ± 6.99 | 72.69 ± 8.32 | 1.73 ± 0.09 | 5.92 ± 1.10 | 2.76 ± 0.45 |
| ENGINE | Full | 70.87 ± 6.99 | 69.19 ± 8.17 | 71.47 ± 9.77 | 6.11 ± 5.22 | 5.40 ± 2.11 | 1.34 ± 0.39 |
| OFA | Full | 38.13 ± 17.05 | 27.43 ± 19.75 | 36.03 ± 16.45 | 3.21 ± 0.06 | 11.25 ± 6.27 | 2.41 ± 1.80 |
| GraphGPT | Adapter | 54.50 ± 12.72 | 58.21 ± 10.61 | 52.74 ± 10.63 | 18.43 ± 0.97 | 130.75 ± 20.89 | 24.29 ± 8.01 |
| LLaGA | Adapter | 60.03 ± 9.34 | 55.48 ± 9.21 | 57.90 ± 10.93 | 18.52 ± 0.65 | 65.69 ± 10.94 | 12.69 ± 5.45 |
| GPF-Plus | Prompt | 60.03 ± 6.93 | 54.57 ± 5.96 | 56.56 ± 7.54 | 1.70 ± 0.06 | 3.91 ± 0.39 | 0.89 ± 0.08 |
| GPrompt | Prompt | 66.10 ± 1.61 | 64.07 ± 2.68 | 66.24 ± 4.56 | 1.74 ± 0.12 | 5.14 ± 1.23 | 1.13 ± 0.11 |
| OpenGraph | Full | 41.93 ± 7.04 | 39.92 ± 6.81 | 42.31 ± 9.71 | 1.68 ± 0.16 | 2.52 ± 1.24 | 0.93 ± 0.15 |
| RiemannGFM | Full | 73.23 ± 7.07 | 69.92 ± 7.36 | 72.46 ± 7.33 | 7.50 ± 1.76 | 15.63 ± 2.98 | 3.88 ± 1.95 |
| GIT | Full | 69.80 ± 2.99 | 68.03 ± 5.02 | 71.09 ± 7.80 | 1.73 ± 0.07 | 1.51 ± 0.12 | 0.79 ± 0.15 |
| SAMGPT | Prompt | 65.00 ± 10.36 | 63.07 ± 10.70 | 65.89 ± 12.10 | 2.12 ± 0.04 | 5.56 ± 0.98 | 3.08 ± 0.52 |
| GraphAny | Full | 49.27 ± 5.01 | 45.15 ± 4.55 | 50.27 ± 4.44 | 2.81 ± 1.33 | 22.28 ± 3.32 | 18.93 ± 3.31 |
| AnyGraph | Full | 54.20 ± 8.17 | 49.78 ± 13.30 | 53.84 ± 13.28 | 2.46 ± 0.18 | 56.84 ± 50.46 | 50.13 ± 47.12 |

### Inference, memory, and updated parameters

| Method | Inference (ms/test graph) | Throughput (nodes/s) | Train peak (GiB) | Inference peak (GiB) | Trainable parameters (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GCN | 2.31 ± 0.37 | 439,365 ± 64,053 | 0.61 ± 0.10 | 0.61 ± 0.10 | 49.87 ± 0.10 |
| GAT | 2.61 ± 0.24 | 384,872 ± 33,879 | 0.68 ± 0.18 | 0.60 ± 0.10 | 100.00 ± 0.00 |
| ENGINE | 129.58 ± 16.72 | 7,807 ± 1,044 | 0.59 ± 0.01 | 0.59 ± 0.01 | 100.00 ± 0.00 |
| OFA | 662.45 ± 101.83 | 1,534 ± 242 | 1.39 ± 0.27 | 1.25 ± 0.24 | 100.00 ± 0.00 |
| GraphGPT | 176,610.72 ± 21,584.61 | 6 ± 1 | 23.65 ± 0.03 | 16.08 ± 0.04 | 0.05 ± 0.00 |
| LLaGA | 146,675.21 ± 22,673.26 | 7 ± 1 | 19.55 ± 0.17 | 16.02 ± 0.19 | 0.13 ± 0.00 |
| GPF-Plus | 77.27 ± 14.36 | 13,281 ± 2,740 | 0.60 ± 0.03 | 0.63 ± 0.08 | 19.63 ± 0.26 |
| GPrompt | 74.64 ± 13.21 | 13,707 ± 2,629 | 0.57 ± 0.03 | 0.60 ± 0.07 | 0.19 ± 0.00 |
| OpenGraph | 40.39 ± 41.71 | 44,736 ± 29,101 | 1.28 ± 0.27 | 1.17 ± 0.34 | 100.00 ± 0.00 |
| RiemannGFM | 351.94 ± 58.53 | 2,891 ± 451 | 2.10 ± 0.66 | 2.57 ± 2.29 | 100.00 ± 0.00 |
| GIT | 2.12 ± 0.76 | 507,436 ± 150,621 | 0.65 ± 0.13 | 0.63 ± 0.12 | 100.00 ± 0.00 |
| SAMGPT | 1.40 ± 0.21 | 724,090 ± 120,865 | 0.78 ± 0.30 | 0.66 ± 0.15 | 0.49 ± 0.00 |
| GraphAny | 0.99 ± 0.01 | 1,011,628 ± 14,427 | 0.52 ± 0.00 | 0.52 ± 0.00 | 100.00 ± 0.00 |
| AnyGraph | 0.50 ± 0.04 | 2,018,820 ± 174,387 | 1.92 ± 0.16 | 1.07 ± 0.39 | 100.00 ± 0.00 |

### Model and artifact size

| Method | Total parameters (M) | Trainable parameters (M) | Saved update/checkpoint (MiB) | Retained artifacts (MiB) | Batch size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GCN | 0.133 | 0.066 | 0.52 | 0.54 | N/A |
| GAT | 0.067 | 0.067 | 0.26 | 0.29 | N/A |
| ENGINE | 1.186 | 1.186 | 4.53 | 4.56 | 256 |
| OFA | 24.205 | 24.205 | 92.41 | 92.43 | 64 |
| GraphGPT | 8,034.460 | 4.198 | 16.02 | 16.13 | 4 |
| LLaGA | 8,040.753 | 10.492 | 40.03 | 40.12 | 4 |
| GPF-Plus | 0.082 | 0.016 | 0.32 | 0.34 | 128 |
| GPrompt | 0.066 | 0.000 | 0.25 | 0.28 | 128 |
| OpenGraph | 25.190 | 25.190 | 96.11 | 96.14 | N/A |
| RiemannGFM | 0.260 | 0.260 | 1.03 | 1.05 | 64 |
| GIT | 0.397 | 0.397 | 1.53 | 1.55 | N/A |
| SAMGPT | 0.234 | 0.001 | 0.91 | 0.94 | N/A |
| GraphAny | 0.003 | 0.003 | 0.02 | 0.04 | 128 |
| AnyGraph | 16.876 | 16.876 | 64.47 | 81.47 | 4,096 |

### Repeated-use time

Each value is cumulative measured reuse time divided by repeated target-only time for N downstream uses. Values below 1 favor reuse.

| Method | Update | N=1 | N=5 | N=10 | N=50 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| GCN | Full | 2.11 | 1.19 | 1.08 | 0.99 |
| GAT | Full | 2.69 | 1.78 | 1.67 | 1.57 |
| ENGINE | Full | 2.48 | 1.36 | 1.22 | 1.11 |
| OFA | Full | 1.36 | 0.79 | 0.71 | 0.66 |
| GraphGPT | Adapter | 1.31 | 0.94 | 0.90 | 0.86 |
| LLaGA | Adapter | 1.25 | 0.94 | 0.90 | 0.87 |
| GPF-Plus | Prompt | 2.30 | 1.35 | 1.23 | 1.13 |
| GPrompt | Prompt | 2.41 | 1.63 | 1.53 | 1.46 |
| OpenGraph | Full | 1.69 | 0.98 | 0.89 | 0.82 |
| RiemannGFM | Full | 2.12 | 1.19 | 1.07 | 0.98 |
| GIT | Full | 1.55 | 0.82 | 0.73 | 0.66 |
| SAMGPT | Prompt | 2.46 | 1.57 | 1.46 | 1.37 |
| GraphAny | Full | 2.17 | 1.19 | 1.07 | 0.97 |
| AnyGraph | Full | 2.77 | 1.34 | 1.16 | 1.02 |

---

## Citation

```bibtex
@misc{opengfm_repo,
  author       = {{OpenGFM contributors}},
  title        = {{OpenGFM}: Experimental Results and Reproduction Materials},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/relaps-e/OpenGFM-Appendix}
}
```
