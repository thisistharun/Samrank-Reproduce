# SamRank Reproduce
To reproducing the paper "[SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2](https://aclanthology.org/2023.emnlp-main.630)" 


## Requirements
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing). Download the zip file and extract it
- Open the folder standford-corenlp-full-2018-02-27 in your IDE and run the following command
  ```shell
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000
  
- Download the dependencies using the following command
  ```shell
  pip install -r requirements.txt

## Running the Code.
- Run the datasets {Inspec/SemEval2010/SemEval2017] using [BERT/GPT2]
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
## Results
- The results can be observed in .csv files in the folder 'experiment_results'
