# Academic References — Experts-First MoE Expert Placement

本プロジェクトの設計根拠となった学術論文一覧。

---

## 直接的根拠

### 1. CoX-MoE (DAC '26)
- **arXiv**: 2605.17889
- **著者**: Muyoung Son, Yi Chen, Seungjae Yoo, Soongyu Choi, Joo-Young Kim
- **採択**: DAC 2026
- **主要手法**: Static expert-aware stratification — 頻繁に選択されるexpertをGPUに事前配置
- **効果**: FlexGen比 7.1x スループット
- **本プロジェクトとの対応**: Phase A (full-slot static preload) の理論的裏付け

### 2. ReMoE (ICML 2026)
- **arXiv**: 2605.27081
- **著者**: Xiongwei Zhu, Xiaojian Liao, Tianyang Jiang, Yusen Zhang, Liang Wang, Limin Xiao
- **採択**: ICML 2026
- **主要手法**: Router fine-tuningでexpert再利用率を向上
- **効果**: llama.cpp上で 1.77-1.99x 高速化 (Jetson Orin NX)
- **本プロジェクトとの対応**: llama.cpp上でのMoE offloading実証

### 3. WiSP (Preprint)
- **arXiv**: 2606.21868
- **著者**: Jiamu Zhang, Liang Wu, Mayank Darbari, Liangjie Hong
- **主要手法**: Working-set paging for MoE — routing-aware expert pager
- **効果**: Static offload比 1.95x decode throughput
- **本プロジェクトとの対応**: frequency配置のrouting-aware設計

### 4. RFID-MoE (Preprint)
- **arXiv**: 2602.09316
- **著者**: Zhendong Mi, Yixiao Chen, Pu Zhao, Xiaodong Yu, Hao Wang, Yanzhi Wang, Shaoyi Huang
- **主要手法**: Routing frequency + information density でexpert重要度を評価
- **効果**: Qwen3-30Bで perplexity 8.0改善
- **本プロジェクトとの対応**: frequency-based placement の理論的根拠

---

## 補強的根拠

### 5. TriMoE (DAC '26)
- **arXiv**: 2603.01058
- **著者**: Yudong Pan et al.
- **採択**: DAC 2026
- **主要知見**: hot/warm/cold の3層expert割り当て → hot=GPU, warm=NDP, cold=CPU
- **効果**: SOTA比 2.83x
- **本プロジェクトとの対応**: expertの温度に基づく配置戦略

### 6. MELINOE (Preprint)
- **arXiv**: 2602.11192
- **著者**: Arian Raje, Anupam Nayak, Gauri Joshi
- **主要手法**: Fine-tuningで少数expertへの集中を促進
- **効果**: 1.2-3x throughput
- **本プロジェクトとの対応**: expert再利用率の向上

### 7. DALI (Preprint)
- **arXiv**: 2602.03495
- **著者**: Zeyu Zhu, Gang Li, Peisong Wang et al.
- **主要手法**: Workload-aware offloading — 0-1整数最適化でCPU/GPU分割
- **効果**: Prefill/Decode両方で高速化
- **本プロジェクトとの対応**: 動的CPU/GPU割り当て

### 8. MoBiLE (ASP-DAC '26)
- **arXiv**: 2510.12357
- **著者**: Yushu Zhao et al.
- **採択**: ASP-DAC 2026
- **主要手法**: 重要度に応じてexpert数を動的変更 (big-little experts)
- **効果**: 1.60-1.72x
- **本プロジェクトとの対応**: expert数の動的調整

### 9. Speculating Experts (Preprint)
- **arXiv**: 2603.19289
- **著者**: Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda
- **主要手法**: 内部表現から次expertを予測してprefetch
- **効果**: TPOT 14%削減
- **本プロジェクトとの対応**: expert prefetching

---

## その他の関連論文

### KTransformers (SOSP 2025)
- De facto standard for MoE inference on consumer hardware
- Prefill 4.62-19.74x speedup
- CPU+GPU collaborative inference

### MoE-APEX (ASPLOS 2026)
- HOBBIT successor, implemented on llama.cpp
- 1.34-9.75x speedup

### SMOE (IPDPS 2026)
- Consumer GPU (671B model)
- 8.68x prefill speedup

### MoE-SpeQ (Preprint)
- PCIe speculative execution
- 2.34x speedup

---

## 本プロジェクトの設計と論文の対応関係

| プロジェクト設計 | 該当論文 | 一致性 |
|----------------|---------|--------|
| Expert選択頻度の収集 | CoX-MoE, RFID-MoE | ✓ 完全一致 |
| frequency順にGPU配置 | CoX-MoE (static stratification) | ✓ 完全一致 |
| VRAM予算内の配置計画 | WiSP (MV-WSA), DALI | ✓ 完全一致 |
| full-slot → partial → dynamic | CoX-MoE → WiSP → DALI | ✓ 段階的発展と一致 |
| GPU/CPU分岐 (compute_tensor) | CoX-MoE, TriMoE | ✓ 完全一致 |

---

## 参考リンク

- CoX-MoE: https://arxiv.org/abs/2605.17889
- ReMoE: https://arxiv.org/abs/2605.27081
- WiSP: https://arxiv.org/abs/2606.21868
- RFID-MoE: https://arxiv.org/abs/2602.09316
- TriMoE: https://arxiv.org/abs/2603.01058
- MELINOE: https://arxiv.org/abs/2602.11192
- DALI: https://arxiv.org/abs/2602.03495
- MoBiLE: https://arxiv.org/abs/2510.12357
- Speculating Experts: https://arxiv.org/abs/2603.19289
