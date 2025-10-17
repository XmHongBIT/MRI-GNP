# MRI-GNP: MRI-based Glioma Neuropathology Prediction

MRI-GNP (MRI-based Glioma Neuropathology Prediction) is a state-of-the-art deep learning system designed for the non-invasive, preoperative prediction of neuropathological features in adult-type diffuse gliomas (ADGs) using standard MR images.Developed using a massive dataset of 35,616 MR images from 8,844 patients across 22 datasets from three countries, this system aims to enhance diagnostic precision and provide crucial support for clinical decision-making.

Core Features

🧠 Comprehensive Neuropathology Prediction: Predicts 12 critical neuropathology markers preoperatively, covering key WHO grading criteria, molecular markers, and genetic alterations.
📄 Virtual Pathology Reports: Integrates model outputs into intuitive "virtual pathology reports," which have been validated to significantly improve the diagnostic accuracy of neuroradiologists.
🧩 Robustness to Missing Data: Incorporates generative models to address the common clinical challenge of missing contrast-enhanced T1-weighted imaging sequences, enhancing the system's real-world applicability.
🌍 High Generalizability: Built on a large-scale, multi-center, and international dataset to ensure the model is robust and generalizable across diverse clinical settings.

Model PerformanceMRI-GNP demonstrates strong and reliable performance across several key prediction tasks. Performance was assessed using the area under the receiver operating characteristic curve (AUC).

Prediction TaskPerformance LevelAUC ScoreHigh WHO GradeHigh0.852IDH MutationHigh0.8261p/19q CodeletionHigh0.823Ki-67 ExpressionHigh0.817WHO Grade 2/3/4Moderate≥ 0.7+7/-10 AlterationModerate≥ 0.7CDKN2A/B Homozygous DeletionModerate≥ 0.7TERT Promoter MutationModerate≥ 0.7EGFR AmplificationModerate≥ 0.7MGMT Promoter MethylationPoor—TP53 MutationPoor—ATRX MutationPoor—

Technical Architecture

After comprehensive evaluation of various deep learning architectures, input formats, and training strategies, the optimal configuration was identified as a Pretrained Vision Transformer (ViT) with a 2.5D input configuration. This approach allows the model to efficiently capture spatial and contextual features from multi-sequence MRI scans.

Clinical Significance

MRI-GNP is a powerful tool with strong potential to enhance patient care by:Enhancing Precision Diagnostics: Providing crucial molecular and genetic information non-invasively before surgery, enabling more accurate diagnosis and tumor classification.Guiding Treatment Planning: Assisting clinicians in tailoring personalized treatment strategies, including surgical extent, radiation, and chemotherapy regimens.Supporting Clinical Decision-Making: Empowering clinical teams with reliable, data-driven insights to support complex decision-making processes and improve patient outcomes.

Installation & Usage (Placeholder)To get started with MRI-GNP, clone the repository and install the required dependencies.Bash# Clone this repository
git clone https://github.com/XmHongBIT/MRI-GNP.git

# Navigate to the project directory
cd MRI-GNP

# Install dependencies
pip install -r requirements.txt

Please refer to the documentation in the docs/ folder for detailed instructions on usage and examples.

How to Contribute (Placeholder)

We welcome contributions of all kinds! If you are interested in improving MRI-GNP, please feel free to fork the repository, submit a pull request, or open an issue.Citation (Placeholder)If you use MRI-GNP in your research, please cite our paper:[under review]

License (Placeholder)This project is licensed under the MIT License.


