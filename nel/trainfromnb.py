# coding: utf-8
import findspark
import pyspark
import os
import pprint
from sift.corpora import wikipedia, wikidata
from sift.models import text, links

from nel.model import data
from nel.model.store import file
from nel.doc import Doc
from nel.harness.format import from_sift
from nel.process.pipeline import Pipeline
from nel.process.candidates import NameCounts
from nel.features.probability import EntityProbability, NameProbability
from nel.learn import ranking
from nel.features import meta
from nel.model import resolution
from nel.process import resolve
from nel.harness.format import inject_markdown_links
from IPython.display import display, Markdown
from nel.process import tag, coref

# load and create a spark context
findspark.init('/home/rohit/spark-2.2.1-bin-hadoop2.7')
sc = pyspark.SparkContext()
sqlContext = pyspark.sql.SQLContext(sc)

wikipedia_base_path = '/home/rohit/wikipedia/dumps/3046511'
# wikidata_base_path = '/n/schwa11/data0/linking/wikidata/dumps/20150713'

# this 'calls' an instance of WikipediaCorpus, then builds the corpus,
# a dictionary with keys id, pid, namespace, content and redirect.
wikipedia_corpus = wikipedia.WikipediaCorpus()(sc, wikipedia_base_path)
docs = wikipedia.WikipediaArticles()(wikipedia_corpus).cache()

wikipedia_pfx = 'en.wikipedia.org/wiki/'

ec_model = links.EntityCounts(min_count=5, filter_target=wikipedia_pfx).build(docs)\
            .map(links.EntityCounts.format_item)

enc_model = links.EntityNameCounts(lowercase=True, filter_target=wikipedia_pfx)\
            .build(docs).filter(lambda (name, counts): sum(counts.itervalues()) > 1)\
            .map(links.EntityNameCounts.format_item)


os.environ['NEL_DATASTORE_URI'] = 'file:///home/rohit/data0/nel/'
# we can use model.toLocalIterator if models don't fit in memory
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
#print(sample)
# clear existing links
for doc in sample:
    for chain in doc.chains:
        chain.resolution = None
        for mention in chain.mentions:
            mention.resolution = None

linked_sample = [linking_pipeline(doc) for doc in sample]
print([d.id for d in linked_sample])
print(sample[0].chains[0].resolution.id)

# int(display(Markdown(inject_markdown_links(linked_sample[0].text, linked_sample[0]))))

mention_detection = [tag.SpacyTagger(), coref.SpanOverlap()]
full_pipeline = Pipeline(mention_detection + candidate_generation + feature_extraction + linking)
linked_sample = [full_pipeline(doc) for doc in sample]
# int(display(Markdown(inject_markdown_links(linked_sample[0].text, linked_sample[0], 'https://'))))