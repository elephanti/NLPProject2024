# Lambada+

This repository contains the code and resources for the **Lambada+** project, developed as the final project report for the NLP course at Reichman University (RUNI) in 2024.

### Project Overview

This project enhances the LAMBADA method by integrating newer models and incorporating label descriptions into the model input to improve the quality of generated synthetic data. The primary goal is to boost model accuracy and robustness, reduce overfitting, and enable NLP systems to better handle diverse real-world scenarios.

The **LAMBADA+** method achieves these improvements through the following five key steps:

1. **Description Generation**: Generate label descriptions using OpenAI’s GPT-4 API and append them to each sample in the dataset, creating an augmented dataset.
2. **Baseline Classifier Training**: Train a baseline classifier using the augmented dataset, following the approach used in the original LAMBADA method. This classifier is later used to filter the synthesized data.
3. **Fine-tuning the LLM**: Fine-tune the LLM to generate data samples based on the label descriptions, resulting in a fine-tuned model.
4. **Data Synthesis**: Use the fine-tuned LLM to generate new text samples for each label and description, creating a diverse dataset.
5. **Data Filtering**: Filter the generated dataset using the baseline classifier, selecting the top \(k\) high-confidence samples to form the final augmented dataset.

### Key Files and Directories

	classifiers/: Contains Python scripts for training classifiers such as BERT, SVM, and LSTM on various datasets.
	datasets/: Contains the ATIS and TREC datasets, including full, test and validation datasets and their respective sampled subsets.
    generated_datasets/: Contains the datasets generated using the fine-tuned LLM models.
    datasets_predictions/: Stores predictions made by the classifiers on the generated datasets.
	filtered_datasets/: Contains filtered versions of the generated datasets, which were processed for higher quality.
	final_results/: Directory intended for storing the final output results.

The main scripts and notebooks for data processing, augmentation, and visualization are located in the root directory.

### Experiments

The experiments in this project evaluated the effectiveness of the LAMBADA+ method compared to the original LAMBADA and a baseline across two datasets: ATIS and TREC. The tests involved classifiers such as BERT, SVM (with TF-IDF and GloVe), and LSTM. The project demonstrated varying degrees of improvement, particularly in scenarios with smaller datasets.

### Experiment Steps

1. **Data Preprocessing**: Use `Dataset_Processing.ipynb` to preprocess the data and generate label descriptions.
2. **Data Augmentation**: Use the `LAMBADA_Data_Augmentation.ipynb` and `LAMBADA+_Data_Augmentation.ipynb` notebooks to generate synthetic datasets.
3. **Baseline Classifier Training**: Run the script `classifiers/classifiers_train.py` to train classifiers on both full and subset datasets and to create predictions for the generated dataset.
4. **Filter the Generated Dataset**: Use the `Filter_datasets.ipynb` notebook to filter the generated dataset using the baseline classifier, selecting the top \(k\) high-confidence samples to form the final augmented dataset.
5. **Training and Testing Classifiers**: Use the `Train_and_Test_with_augmented_data.ipynb` notebook to train and test classifiers on both the original and augmented datasets.