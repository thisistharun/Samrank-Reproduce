# SamRank Reproduce
"[SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2](https://aclanthology.org/2023.emnlp-main.630)" 

### To Reproduce our Results


## Requirements
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing). Download the zip file and extract it
- Open the folder standford-corenlp-full-2018-02-27 in your IDE and run the following command
  ```
  cd standford-corenlp-full-2018-02-27
  ```

  ```
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000


- Download the dependencies using the following command
  ```
  pip install -r requirements.txt

## Running the Code.
- For running in Both Global and Propotional Modes
- Select any one of the dataset from data folder and one model from [BERT/GPT2/mBERT/RoBERTa] to get the results in both modes.
- Repeat for other datasets to evaluate the model
- Follow the below mentioned example format to execute the code
```shell
python samrank.py --dataset English_Inspec --plm BERT
```
- For running in Individual Modes
- Select any one of the dataset from data folder and one model from [BERT/GPT2/mBert/RoBERTa] along with any of the mode [Global/Propotional]
- Repeat for other datasets to evaluate the model
- Follow the below mentioned example format to execute the code
```shell
python samrank.py --dataset English_Inspec --plm GPT2 --mode Global

```
## Results
- The results can be observed in .csv files in the folder 'experiment_results'.
- Also Top 3 Heads for f1@5, f1@10 and f1@15 are displayed on terminal after execution.
