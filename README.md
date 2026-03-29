# OpenGFM: A Unified Evaluation Framework for Graph Foundation Models

This repository contains the official implementation, environment setups, exhaustive hyperparameter search spaces, and complete prompt templates for the paper **"Graph Foundation Models: Taxonomy, Experimental Study and Future Directions"**.

Due to space limits in the main text, this repository serves as the supplementary material, providing comprehensive details to ensure full reproducibility and to foster further research in Graph Foundation Models (GFMs).

---

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
| **GNN-Enhanced LLM** | LLaGA | ✅ | ✅ | ❌ | ✅ (Multi-Source) | Pre-trained LLM | In-context |
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

### 3. GNN-Enhanced LLMs
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

