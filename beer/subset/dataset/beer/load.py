import cPickle as pickle
from path import Path
from beer import Beer, Review
from tqdm import tqdm
import logging

def load_json(data_dir, file):
    with open(data_dir / file) as fp:
        for line in fp:
            yield eval(line.strip())

def load_data(data_dir, file='beer_50000.json', cache_file='cache.pkl'):
    data_dir = Path(data_dir)
    if (data_dir / cache_file).exists():
        logging.info("Loading beer data from cache...")
        with open(data_dir / cache_file, 'rb') as fp:
            reviews, beers = pickle.load(fp)
    else:
        logging.info("Loading beer data from file...")

        beers = {}
        reviews = []
        for review in tqdm(load_json(data_dir, file)):
            beer_id = int(review['beer/beerId'])
            if beer_id not in beers:
                try:
                    beers[beer_id] = Beer.from_json(review)
                except:
                    continue
            beer = beers[beer_id]

            reviews.append(Review.from_json(review, beer))
        with open(data_dir / cache_file, 'wb') as fp:
            pickle.dump((reviews, beers), fp)
    return reviews, beers
