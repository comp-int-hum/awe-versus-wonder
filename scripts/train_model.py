import argparse
import logging
import json
import re
import csv
import pickle
import math
import random
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pandas


logger = logging.getLogger("train_model")


def split_doc(tokens, max_length):
    num_subdocs = math.ceil(len(tokens) / max_length)
    retval = []
    for i in range(num_subdocs):
        retval.append(tokens[i * max_length : (i + 1) * max_length])
    return retval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Excel file of application data")
    parser.add_argument("--model_output", dest="model_output", help="File to save model to", required=True)
    parser.add_argument("--topic_output", dest="topic_output", help="File to write topic summary to", required=True)
    parser.add_argument("--annotated_output", dest="annotated_output", help="File to save annotated input to", required=True)
    parser.add_argument("--max_doc_length", dest="max_doc_length", type=int, default=200, help="Documents will be split into at most this length for training (this determines what it means for words to be 'close')")
    parser.add_argument("--num_topics", dest="num_topics", type=int, default=50, help="How many topics the model will infer")
    parser.add_argument("--min_word_length", dest="min_word_length", type=int, default=4, help="Words shorter than this will be ignored")
    parser.add_argument("--min_word_occurrence", dest="min_word_occurrence", type=int, default=2, help="Words occuring less than this number of times throughout the entire dataset will be ignored")
    parser.add_argument("--max_word_proportion", dest="max_word_proportion", type=float, default=0.5, help="Words occuring in more than this proportion of documents will be ignored (probably conjunctions, etc)")    
    
    parser.add_argument("--top_words", dest="top_words", type=int, default=10, help="Number of words to show for each topic in the summary file")
    parser.add_argument("--iterations", dest="iterations", type=int, default=10, help="How long to train")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.random_seed:
        random.seed(args.random_seed)
    
    data = pandas.read_excel(args.input)
    lemmatizer = WordNetLemmatizer()
    
    orig_docs = {}
    for i, row in data.iterrows():
        for k, v in row.items():
            if "_question" in str(k) and len(str(v)) > 50 and not k.startswith("interview"):
                orig_docs[(i, k)] = [
                    (t, lemmatizer.lemmatize(re.sub(r"[^a-z]", "", t.lower())))
                    for t in re.split(r"\s+", v)
                ]

    logger.info("Found %d documents that are probably candidate-essays", len(orig_docs))
    
    train_docs = [[tok for _, tok in doc if len(tok) >= args.min_word_length] for doc in orig_docs.values()]
    train_subdocs = sum([split_doc(doc, args.max_doc_length) for doc in train_docs], [])
    
    logger.info("Created %d subdocs of maximum length %d tokens", len(train_subdocs), args.max_doc_length)
    
    dictionary = Dictionary(train_docs)
    dictionary.filter_extremes(no_below=args.min_word_occurrence, no_above=args.max_word_proportion)
    
    corpus = [dictionary.doc2bow(subdoc) for subdoc in train_subdocs]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=args.num_topics,
        alpha="auto",
        eta="auto",
        iterations=args.iterations,
        passes=args.iterations,
        eval_every=None,
        random_state=args.random_seed
    )

    logger.info("Saving trained model to '%s'", args.model_output)
    with open(args.model_output, "wb") as ofd:
        ofd.write(pickle.dumps(model))

    logger.info("Writing top %d words and their probabilities for each topic to '%s'", args.top_words, args.topic_output)
    with open(args.topic_output, "wt") as ofd:
        c = csv.writer(ofd, delimiter="\t")
        c.writerow(["Topic #"] + ["Rank {} word".format(i + 1) for i in range(args.top_words)])
        for i, topic in enumerate(model.top_topics(corpus, topn=args.top_words)):
            c.writerow([i] + [w for _, w in topic[0]])
            c.writerow([""] + ["{:.3f}".format(p) for p, _ in topic[0]])

    logger.info("Writing annotated documents to '%s'", args.annotated_output)

    with open(args.annotated_output, "wt") as ofd:
        for (row_num, field_name), doc in orig_docs.items():
            doc_bow = model.id2word.doc2bow([x for _, x in doc])
            doc_bow_lookup = dict(doc_bow)
            _, assignments, _ = model.get_document_topics(
                doc_bow,
                per_word_topics=True
            )
            assignments = dict(assignments)
            annotated = [
                (t, assignments.get(model.id2word.token2id.get(lt), [])) for t, lt in doc
            ]
            ofd.write(
                json.dumps(
                    {
                        "row" : row_num,
                        "field" : field_name,
                        "topic_assignments" : [(t, None if len(topics) == 0 else topics[0]) for t, topics in annotated]
                    }
                ) + "\n"                
            )
