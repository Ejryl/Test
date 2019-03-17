import copy


class HashtagsInfo():
    def __init__(self, text):
        self.text_array = text.split()

        self.stats = self.extract_hashtags()

    def extract_hashtags(self):
        def is_hashtag(w):
            return w[0] == "#"
        current_hashtags = list()
        dist = -1
        dicto = {}
        for w in self.text_array:
            dist += 1
            if is_hashtag(w):
                if dist == 1:
                    current_hashtags.append(w)
                else:
                    current_hashtags = [w]
                dist = 0
            else:
                if w not in dicto:
                    dicto[w] = {}
                for hashtag in current_hashtags:
                    if hashtag not in dicto[w]:
                        dicto[w][hashtag] = []
                    dicto[w][hashtag].append(dist)
        return dicto
    def get_frequencies(self):
        d0 = copy.deepcopy(self.stats)
        d2 = d0
        for w in d0:
            d1 = d0[w]
            total = 0
            for hashtag in d1:
                d2[w][hashtag] = len(d1[hashtag])
                total += d2[w][hashtag]
            for hashtag in d2[w]:
                d2[w][hashtag] = d2[w][hashtag] / total
        return d2

    def get_words_score(self):
        d_hashtags = {}
        for w in self.stats:
            for hashtag in self.stats[w]:
                if hashtag not in d_hashtags:
                    d_hashtags[hashtag] = {}
                d_hashtags[hashtag][w] = len(self.stats[w][hashtag])
        return d_hashtags

    def get_top_words(self):
        d = self.get_words_score()
        d3 = {}
        for hashtag in d:
            s = [k for k in sorted(d[hashtag], key=d[hashtag].get, reverse=True)]
            d3[hashtag] = s
        return d3
text = "#japan #us the japan is a country eastern asia , #uk united kingdom (usage) is not in european union , #japan japan is actually using yen , #japan yen is official japan money "
hashtags_info = HashtagsInfo(text)
frequencies = hashtags_info.get_frequencies()
top_words = hashtags_info.get_top_words()
exit()









        

