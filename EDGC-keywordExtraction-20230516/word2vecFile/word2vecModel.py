#-*- encoding:utf-8 -*-
from function.keyWordExtract import WordSegmentation
from gensim.models import Word2Vec
import os


wordSeg = WordSegmentation(stop_words_file="../textrank4zh/stopwords.txt")

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    return corpus

def segment_sentences(sentences):
    segmented_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        words = wordSeg.segment(sentence)
        segmented_sentence = [word for word in words]
        segmented_sentences.append(segmented_sentence)
    return segmented_sentences

def save_model():
    dataFilterData = read_corpus("./dataFile/dataFilterData.txt")
    segSentences = segment_sentences(dataFilterData)
    model = Word2Vec(segSentences, min_count=3, vector_size=256, window=5, workers=4)
    model.save('./model/word2vec.model')


if __name__=="__main__":

    model_path = './model/word2vec.model'
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        # 使用模型，例如获取某个单词的词向量
        vector = model.wv['密码机']
        print("Vector representation of 'your_word' is: ", vector)

    else:
        save_model()









