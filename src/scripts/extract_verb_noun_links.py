import argparse
import pickle
import pandas as pd

from gulpio2 import GulpDirectory
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Extract verb-noun links from a given dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("verb_noun_pickle", type=Path, help="Path to pickle file to save verb-noun links")
parser.add_argument("verb_classes", type=Path, help="Path to verb classes csv")
parser.add_argument("noun_classes", type=Path, help="Path to noun classes csv")

def main(args):

    verbs = pd.read_csv(args.verb_classes)
    nouns = pd.read_csv(args.noun_classes)

    verb_noun_unique = {}

    dataset = GulpDirectory(args.gulp_dir)

    for i, c in tqdm(
        enumerate(dataset),
        unit=" chunk",
        total=dataset.num_chunks,
        dynamic_ncols=True
    ):
        for _, batch_labels in c:
            if verbs['key'][batch_labels['verb_class']] in verb_noun_unique:
                verb_noun_unique[verbs['key'][batch_labels['verb_class']]].append(nouns['key'][batch_labels['noun_class']])
            else:
                verb_noun_unique[verbs['key'][batch_labels['verb_class']]] = [nouns['key'][batch_labels['noun_class']]]

    with open(args.verb_noun_pickle, 'wb') as f:
        pickle.dump({
            verb: list(set(verb_noun_unique[verb])) for verb in verb_noun_unique.keys()
        }, f)

if __name__ == '__main__':
    main(parser.parse_args())
