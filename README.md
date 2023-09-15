# Topic modeling (towards investigating awe and wonder)

This repository consists of a single script that trains a topic model on data drawn from an Excel spreadsheet with particular structure, applies the model back to the spreadsheet, and saves 1) the model, 2) a summary of the inferred topics, and 3) the token-level topic assignments with information denoting the row and column of the original text.

To prepare to run the code, make sure you have a recent version of Python installed, clone this repository, from a command line with the repository as its working directory, invoke:

```
python3 -m venv local
source local/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

Then, to run the script, invoke:

```
python scripts/train_model.py --input my_excel_file.xls --model_output model.bin --topic_output topics.tsv --annotated_output annotated_text.jsonl
```

Replacing the first file name with your spreadsheet, and the others with whatever file names you want for the outputs.  The outputs will be the serialized model (it can then be loaded into another script if desired, to inspect, annotate more documents, etc), a *truncated summary* of each inferred topic showing *just a few of its top words* (this is in tab-separated format), and the full annotation of each essay as one JSON dictionary per line, where each dictionary has the fields "row" (the row number in the original Excel file), "field" (the name of the column in the original Excel file), and "topic_assignments", which is a list of the essay's (lower-cased) words paired with the topic number that produced it (or "null", if the word was too short, frequent, or infrequent for the model to consider).  JSON can be read very easily by all programming languages (or by humans: it's really simple).

Note that the script has several options that can be specified, but otherwise use default values.  You can see the options, and their default values, by running:

```
python scripts/train_model.py -h
```
