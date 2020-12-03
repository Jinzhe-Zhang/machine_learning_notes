from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
if __name__ == '__main__':
    corpus = ['中文 文本 分类 是 自然语言 处理 中 的 一个 基本 问题',
                '我 爱 自然语言 处理',
                '这 是 一个 问题 以前 我 从来 没有 遇到 过']
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    for i in range(len(word)):
        print (word[i])
    weight = tfidf.toarray()
    print (weight)
    for i in range(len(weight)):
        print('第',i+1,'篇文档的词语tf-idf权重：')
        for j in range(len(word)):
            print (word[j],weight[i][j])