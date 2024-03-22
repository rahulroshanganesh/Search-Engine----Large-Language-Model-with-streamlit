import txtai
import numpy as np
import pandas as pd

np.random.seed(1)

#Amazon product review
# df = pd.read_csv('train.csv')

#seth blogs dataset
df = pd.read_csv('seth-data.csv').dropna()
content = df.content_plain.values
# print(df.head())

# titles = df.dropna().sample(100000).TITLE.values

embeddings = txtai.Embeddings({
    'path': 'sentence-transformers/all-MiniLM-L6-v2'
})

# embeddings.load('embeddings.tar.gz')

# embeddings.index(titles)
embeddings.index(content)


# embeddings.save('embeddings.tar.gz')
embeddings.save('embeddings_seth.tar.gz')


# result = embeddings.search(query="protector for cam", limit=5)


# print(result)

# actual_results = [titles[x[0]] for x in result]

# print(actual_results)