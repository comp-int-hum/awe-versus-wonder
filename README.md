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

Here is the first 15 topics from running the script on this study's data, using default parameters and setting the random seed to 0:

|Topic|Word 1|Word 2|Word 3|Word 4|Word 5|Word 6|Word 7|Word 8|Word 9|Word 10|
|---|---|---|---|---|---|---|---|---|---|---|
|0 |      family | parent | were |   friend|  mother|  home |   felt |   they |   year |   life|
|   |     0.040 |  0.023 |  0.022 |  0.016 |  0.016 |  0.013 |  0.011 |  0.011 |  0.010 |  0.010|
|1  |     disease |interest |       cancer|  medicine    |    patient |interested  |    research    |    area  |  field |  treatment|
|   |     0.033 |  0.025 |  0.024 |  0.022  | 0.020 |  0.019 |  0.016 |  0.015 |  0.014 |  0.012|
|2  |     patient| care |   will  |  physician  |     medical |help  |  health | medicine |       these |  future|
|   |     0.050 |  0.027 |  0.027  | 0.026 |  0.024 |  0.017 |  0.014 |  0.013 |  0.012 |  0.011|
|3  |     school | year  |  college |high  |  class  | through| work  |  first |  support |life|
|   |     0.024 |  0.020 |  0.019 |  0.012  | 0.009 |  0.008  | 0.008 |  0.008  | 0.007 |  0.007|
|4  |     could |  class |  would |  after |  thing |  back  |  myself | life |   felt |   even|
|   |     0.017 |  0.016 |  0.015 |  0.013 |  0.011 |  0.010 |  0.010  | 0.009 |  0.009 |  0.009|
|5  |     what  |  would |  they |   know  |  there |  were |   most |   some |   just  |  them|
|   |     0.012 |  0.010 |  0.009 |  0.008 |  0.008 |  0.008 |  0.007 |  0.007 |  0.007 |  0.006|
|6  |     wonder | life |   world |  every |  there  | into |   moment | people | through | around|
|   |     0.015 |  0.012 |  0.010 |  0.009 |  0.009 |  0.009 |  0.009 |  0.009 |  0.009 |  0.008|
|7  |     people | community |      minority |       school | background  |    group  | culture |student |being  | different|
|   |     0.018 |  0.016 |  0.014 |  0.013 |  0.012 |  0.012  | 0.011 |  0.010 |  0.009  | 0.009 |
|8  |     they  |  were  |  them  |  what  |  people | other |  because| others | person | different|
|   |     0.037 |  0.022 |  0.021 |  0.017 |  0.014 |  0.014 |  0.012 |  0.010 |  0.010 |  0.009|
|9 |      student| program| university  |    school | organization  |  community  |     them  |  college |member | group|
|  |      0.067 |  0.016 |  0.011 |  0.011 |  0.010 |  0.009 |  0.007 |  0.007 |  0.007 |  0.006|
|10  |    life |   could |  through| year |   while |  during | learned |which |  father|  support|
|   |     0.012  | 0.011 |  0.010 |  0.010 |  0.009 |  0.009 |  0.008 |  0.008 |  0.007 |  0.007|
|11 |     could |  hour |   after |  research   |     data |   into  |  result | first |  each  |  week|
|   |     0.012 |  0.008 |  0.008 |  0.007 |  0.007  | 0.007 |  0.006 |  0.006 |  0.006 |  0.006|
|12 |     food  |  meal  |  cooking| would |  cook |   eating | ingredient |     family|  others | only|
|  |      0.041 |  0.021 |  0.019 |  0.017 |  0.014 |  0.012 |  0.012 |  0.011 |  0.011 |  0.011|
|13 |     research  |      medicine  |      clinical  |      science |through| hopkins| scientific  |    cell |   opportunity  |   john|
|   |     0.060 |  0.015 |  0.013 |  0.011 |  0.011 |  0.010 |  0.009  | 0.009 |  0.009  | 0.009|
|14 |     team |   year  |  training  |      sport|   myself | would |  coach |  other  | teammate |       were|
|   |     0.043 |  0.015  | 0.014 |  0.013  | 0.012 |  0.010 |  0.010  | 0.009  | 0.008 |  0.008|
