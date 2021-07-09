from textPreprocessing import *
from getData import *
from getUmapEmbeddings import *
from hdbscanClusters import *
from GenerateDataset import *
from gensim.parsing.preprocessing import remove_stopwords
from TopicBokehPlot import *

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Working Directory #
dir = os.getcwd() + r"\\"
data_df = pd.read_csv(dir + 'Data\SampleData.csv')

def main(dataDF):
    """
    dataDF: Dataframe that contains text under a title column
    """
    path_to_model = dir + 'Model\distilbert-base-nli-stsb-mean-tokens'

    data = list(dataDF.title)

    # Model Parameters #
    embeddings = transformerModel(data, path_to_model, 256)
    umap_embeddings = umapEmbeddings(15, 2, 'cosine', embeddings)
    cluster = topicClusters(10, 'euclidean', 'leaf', 0.1, umap_embeddings)

    result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    return result,data
