from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
f = open("transformer-xl-master/data/wikitext-103/train.txt", "r", encoding="utf8")
i = 0
nb_lines = 1000
text = ""
for line in f:
    if i >= nb_lines:
        break
    text += line
    i += 1
regex = "(?:= )+(?:.*?)(?:= )+"
contents = re.split(regex, text)
contents = [x[4:] for x in contents]
contents = [x for x in contents if len(x) > 10]

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(contents)
tf_feature_names = tf_vectorizer.get_feature_names()
no_topics = 20
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10

display_topics(lda, tf_feature_names, no_top_words)
exit(0)