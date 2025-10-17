# MRI-GNP: MRI-based Glioma Neuropathology Prediction Deep Learning System



**MRI-GNP (MRI-based Glioma Neuropathology Prediction)** is a state-of-the-art deep learning system designed for the non-invasive, preoperative prediction of neuropathological features in adult-type diffuse gliomas (ADGs) using standard MR images.

Developed using a massive dataset of **35,616 MR images from 8,844 patients across 22 datasets from three countries**, this system aims to enhance diagnostic precision and provide crucial support for clinical decision-making.

## Core Features

* **🧠 Comprehensive Neuropathology Prediction**: Predicts 12 critical neuropathology markers preoperatively, covering key WHO grading criteria, molecular markers, and genetic alterations.
* **📄 Virtual Pathology Reports**: Integrates model outputs into intuitive "virtual pathology reports," which have been validated to significantly improve the diagnostic accuracy of neuroradiologists.
* **🧩 Robustness to Missing Data**: Incorporates generative models to address the common clinical challenge of missing contrast-enhanced T1-weighted imaging sequences, enhancing the system's real-world applicability.
* **🌍 High Generalizability**: Built on a large-scale, multi-center, and international dataset to ensure the model is robust and generalizable across diverse clinical settings.

## Model Performance

MRI-GNP demonstrates strong and reliable performance across several key prediction tasks. Performance was assessed using the area under the receiver operating characteristic curve (AUC).

| Prediction Task | Performance Level | AUC Score |
| :--- | :---: | :---: |
| **High WHO Grade** | **High** | **0.852** |
| **IDH Mutation** | **High** | **0.826** |
| **1p/19q Codeletion** | **High** | **0.823** |
| **Ki-67 Expression** | **High** | **0.817** |
| WHO Grade 2/3/4 | Moderate | ≥ 0.7 |
| +7/-10 Alteration | Moderate | ≥ 0.7 |
| CDKN2A/B Homozygous Deletion | Moderate | ≥ 0.7 |
| TERT Promoter Mutation | Moderate | ≥ 0.7 |
| EGFR Amplification | Moderate | ≥ 0.7 |
| MGMT Promoter Methylation | Poor | — |
| TP53 Mutation | Poor | — |
| ATRX Mutation | Poor | — |

## Technical Architecture

After comprehensive evaluation of various deep learning architectures, input formats, and training strategies, the optimal configuration was identified as a **Pretrained Vision Transformer (ViT)** with a **2.5D input configuration**. This approach allows the model to efficiently capture spatial and contextual features from multi-sequence MRI scans.

## Clinical Significance

MRI-GNP is a powerful tool with strong potential to enhance patient care by:
* **Enhancing Precision Diagnostics**: Providing crucial molecular and genetic information non-invasively before surgery, enabling more accurate diagnosis and tumor classification.
* **Guiding Treatment Planning**: Assisting clinicians in tailoring personalized treatment strategies, including surgical extent, radiation, and chemotherapy regimens.
* **Supporting Clinical Decision-Making**: Empowering clinical teams with reliable, data-driven insights to support complex decision-making processes and improve patient outcomes.

## Installation & Usage (Placeholder)

To get started with MRI-GNP, clone the repository and install the required dependencies.

```bash
# Clone this repository
git clone https://https://github.com/XmHongBIT/MRI-GNP.git

# Navigate to the project directory
cd MRI-GNP

# Install dependencies
pip install -r requirements.txt

Please refer to the documentation in the docs/ folder for detailed instructions on usage and examples.
```
## How to Contribute (Placeholder)

We welcome contributions of all kinds! If you are interested in improving MRI-GNP, please feel free to fork the repository, submit a pull request, or open an issue.

## Citation (Placeholder)

If you find this work useful in your research, please consider citing our preprint:
```bash
@article{mri_gnp_preprint_2025,
  title={{MRI-Based Deep Learning System for Noninvasive Neuropathological Profiling of Adult-Type Diffuse Glioma}},
  author={Hong, Xiaoming and Li, Yangyang and Li, Yangyang and Xue, Yunjing and Xue, Yunjing and Yang, Ruimeng and Liu, Chenghao and Li, Junjie and Li, Junjie and Pang, Haowen and Shi, Dongli and Shi, Dongli and Liu, Zhaoxi and Liu, Zhaoxi and Qiu, Jun and Qiu, Jun and Jing, Ying and Jing, Ying and Mao, Yu and Mao, Yu and Xu, Siyao and Xu, Siyao and Huang, Xufang and Huang, Xufang and Hua, Tiantian and Hua, Tiantian and Duan, Yunyun and Wu, Minghao and Wu, Minghao and Wang, Jingxuan and Wang, Jingxuan and yuerong, Lizhu and Zhang, Xinru and Liu, Meichen and Jiang, Runze and Zhang, Peng and Barkhof, Frederik and Keil, Vera and Keil, Vera and Zhu, Mingwang and Zhu, Mingwang and Zhang, Zhiqiang and Li, Huan and Li, Huan and Qian, Yingfeng and Qian, Yingfeng and Ma, Heng and Ma, Heng and Li, Xiaodan and Li, Xiaodan and Xu, Rui and Xu, Rui and Zhang, Jing and Zhou, Fuqing and Guo, Jun and Chang, Qing and Zhang, Wei and Zhang, Renlong and Guo, Ya and Meng, Li and Meng, Li and Wang, Guangbin and Wang, Guangbin and Zhuo, Zhi-Zheng and Ye, Chuyang and Liu, Yaou,
  year={2025},
  journal={SSRN Electronic Journal},
  url={[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5552565](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5552565)}
}
```

## License (Placeholder)
This project is licensed under the [MIT License](LICENSE.txt).
