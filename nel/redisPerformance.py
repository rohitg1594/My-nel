# coding: utf-8
import findspark
import pyspark
import os
from datetime import datetime

from sift.corpora import wikipedia, wikidata
from sift.models import text, links
from nel.model import data
from nel.harness.format import from_sift
from nel.process.pipeline import Pipeline
from nel.process.candidates import NameCounts
from nel.features.probability import EntityProbability, NameProbability
from nel.learn import ranking
from nel.features import meta
from nel.process import resolve
from nel.process import tag, coref

# load and create a spark context
findspark.init('/home/rohit/spark-2.2.1-bin-hadoop2.7')
sc = pyspark.SparkContext()
sqlContext = pyspark.sql.SQLContext(sc)

apps = ['app' + str(i) for i in[1,2,4,8]]

start = datetime.now()
wikipedia_base_path = '/home/rohit/bz2/enwiki-latest-pages-articles27.xml-p56163464p56188317'

wikipedia_corpus = wikipedia.WikipediaCorpus()(sc, wikipedia_base_path)
docs = wikipedia.WikipediaArticles()(wikipedia_corpus).cache()

wikipedia_pfx = 'en.wikipedia.org/wiki/'

ec_model = links.EntityCounts(min_count=5, filter_target=wikipedia_pfx).build(docs)\
            .map(links.EntityCounts.format_item)

enc_model = links.EntityNameCounts(lowercase=True, filter_target=wikipedia_pfx)\
            .build(docs).filter(lambda (name, counts): sum(counts.itervalues()) > 1)\
            .map(links.EntityNameCounts.format_item)

data.ObjectStore.Get('models:ecounts[wikipedia]').save_many(ec_model.toLocalIterator())
data.ObjectStore.Get('models:necounts[wikipedia]').save_many(enc_model.toLocalIterator())

candidate_generation = [NameCounts('wikipedia', 10)]
feature_extraction = [EntityProbability('wikipedia'), NameProbability('wikipedia')]
training_pipeline = Pipeline(candidate_generation + feature_extraction)
training_docs = [from_sift(doc) for doc in docs.takeSample(False, 100)]
train = [training_pipeline(doc) for doc in training_docs]

ranker = ranking.TrainLinearRanker(name='ranker', features=[f.id for f in feature_extraction])(train)
classifier_feature = meta.ClassifierScore(ranker)
linking = [classifier_feature, resolve.FeatureRankResolver(classifier_feature.id)]
linking_pipeline = Pipeline(candidate_generation + feature_extraction + linking)
sample = [from_sift(doc) for doc in docs.takeSample(False, 10)]
# clear existing links
for doc in sample:
    for chain in doc.chains:
        chain.resolution = None
        for mention in chain.mentions:
            mention.resolution = None

linked_sample = [linking_pipeline(doc) for doc in sample]
mention_detection = [tag.SpacyTagger(), coref.SpanOverlap()]
full_pipeline = Pipeline(mention_detection + candidate_generation + feature_extraction + linking)
linked_sample = [full_pipeline(doc) for doc in sample]
print([d.id for d in linked_sample])
print(linked_sample)
print(linked_sample[0].chains)
print(datetime.now() - start)



