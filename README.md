# SamRank Reproduce
We are reproducing the paper "[SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2](https://aclanthology.org/2023.emnlp-main.630)" 


## Requirements
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing). Download the zip file and extract it
- Open the folder standford-corenlp-full-2018-02-27 in your IDE and run the following command
  ```shell
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000
  
- Download all the dependencies using the following command
  ```shell
  pip install -r requirements.txt

## Runing
```shell
python samrank.py --dataset Inspec --plm BERT
```
```shell
python samrank.py --dataset Inspec --plm GPT2
```
```shell
python samrank.py --dataset SemEval2010 --plm BERT
```
```shell
python samrank.py --dataset SemEval2010 --plm GPT2
```
```shell
python samrank.py --dataset SemEval2017 --plm BERT
```
```shell
python samrank.py --dataset SemEval2017 --plm GPT2
```

**The performances of all 144 heads** will be saved as data frames (.csv) in the 'experiment_results' folder.
