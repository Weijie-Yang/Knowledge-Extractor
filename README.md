# Fine-tuning Qwen3-8B on HydroBench

This repository provides scripts and data format definitions for fine-tuning **Qwen3-8B** on the **HydroBench** hydrogen-energy Q&A benchmark.

HydroBench is a bilingual (Chinese/English) domain dataset focusing on **hydrogen production, storage, transport, fuel cells, safety, systems, and AI**. It is designed for:

* Teaching / exam question answering
* Domain-specific assistant training
* Evaluation of LLMs on hydrogen energy literacy

---

## 1. Dataset Overview

HydroBench is organized as **bilingual Q&A pairs**, each with metadata describing its module and subtopic.

Each example has (at least) the following fields:

* `id`: unique identifier, e.g. `A-ELC-04`
* `module`: high-level module (A–E)
* `subtopic`: submodule shorthand, e.g. `ELC`
* `question_zh`, `question_en`: question in Chinese / English
* `answer_zh`, `answer_en`: reference answer in Chinese / English
* (optional) other fields such as `difficulty`, `source`, etc., depending on your version

### 1.1 Module Summary

The dataset currently contains **≈648 Q&A pairs** (Chinese + English), distributed as follows:

| Module | Description                                     | #Q&A |
| ------ | ----------------------------------------------- | ---- |
| A      | Hydrogen classification and production          | 135  |
| B      | Storage and transportation                      | 130  |
| C      | Fuel cells and applications                     | 150  |
| D      | Safety, economics, and strategy                 | 148  |
| E      | System integration, AI, and life-cycle analysis | 85   |

### 1.2 Submodule Breakdown

| Submodule | Description                                                              | #Q&A |
| --------- | ------------------------------------------------------------------------ | ---- |
| A-CLS     | Color / classification definitions                                       | 7    |
| A-SMR     | Fossil / blue hydrogen (gasification / reforming)                        | 32   |
| A-ELC     | Water electrolysis (AWE / PEM / SOEC / AEM)                              | 65   |
| A-NEW     | Emerging H₂ production (photo-, electro-, thermo-catalysis, etc.)        | 31   |
| B-GAS     | Gaseous storage & transport (high-pressure tanks / pipelines / blending) | 44   |
| B-LIQ     | Liquid hydrogen                                                          | 42   |
| B-SOLC    | Solid & chemical carriers (ammonia / LOHC / metal hydrides)              | 44   |
| C-TRA     | Transportation applications                                              | 65   |
| C-IND     | Industrial applications                                                  | 45   |
| C-PWR     | Power generation & buildings (CHP, reversible fuel cells)                | 40   |
| D-SAF     | Safety and standards                                                     | 50   |
| D-ECO     | Economics and cost                                                       | 40   |
| D-STR     | Global strategy                                                          | 58   |
| E-AI      | AI and digital twins                                                     | 15   |
| E-INT     | Electricity–hydrogen–carbon system integration                           | 20   |
| E-LCA     | Life cycle assessment (LCA)                                              | 15   |
| E-BUS     | Business models / policy (label name may be adjusted)                    | 15   |
| E-SUP     | Supply chain / supporting systems (label name may be adjusted)           | 20   |

You can select **Chinese only**, **English only**, or **bilingual** samples when constructing training data.

---
