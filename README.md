# Zero-Shot Cross-Domain Slot Filling with Retrieval Augmented In-Context Learning

Requirements:

-	numpy==1.24.4
-	tqdm==4.62.2
-	python==3.8.18
-	pytorch==1.9.0
-	cudatoolkit==10.2.89 
-	transformers==4.10.0
-	sentence-transformers==2.2.2
-	openprompt==1.0

## Files Description
The **/raw_data_snips** contains the snips dataset.

The **retriever.py** can construct data with retrieved examples from other domains (except itself and target domain).

The **reader.py** can read dataset for training, validation and testing.

The **evaluate.py** can evaluate and calcualate average F1 score of slot filling on the snips dataset.

The **config.py** can set the files path and hyper parameters.

The **train_main.py** can perform model training and testing.

## Notes and Acknowledgments
The implementation is based on [https://github.com/LiXuefeng2020ai/GZPL](https://github.com/LiXuefeng2020ai/GZPL)
