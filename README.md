# SamRank Reproduce
To reproduce the paper "[SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2](https://aclanthology.org/2023.emnlp-main.630)" 


## Requirements
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing). Download the zip file and extract it
- Open the folder standford-corenlp-full-2018-02-27 in your IDE and run the following command
  ```shell
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000
  
- Download the dependencies using the following command
  ```shell
  pip install -r requirements.txt

## Running the Code.
- For running in Both Global and Propotional Modes
- Select any one of the dataset from [Inspec/SemEval2010/SemEval2017] and one model from [BERT/GPT2] to get the results in both modes
- Repeat for other datasets to evaluate the model
```shell
python samrank.py --dataset [Inspec/SemEval2010/SemEval2017] --plm [BERT/GPT2]
```
- For running in Individual Modes
- Select any one of the dataset from [Inspec/SemEval2010/SemEval2017] and one model from [BERT/GPT2] along with any of the mode [Global/Propotional]
- Repeat for other datasets to evaluate the model
```shell
python samrank.py --dataset [Inspec/SemEval2010/SemEval2017] --plm [BERT/GPT2] --mode [Global/Propotional]

```
## Results
- The results can be observed in .csv files in the folder 'experiment_results'.
- Also Top 3 Heads for f1@5, f1@10 and f1@15 are displayed on terminal after execution.
