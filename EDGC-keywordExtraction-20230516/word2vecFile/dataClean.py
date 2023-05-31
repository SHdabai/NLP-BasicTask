import re
import jieba
from pprint import pprint
def clean_text(text):
    # 去除HTML标签和CSS样式
    # r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】→《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*∗ +,-./:：γ −∞ ② ——Θ;<=>?@，≥δ·≈ ★、…【】→①《》“”‘’[\\]^_`{|}~「」『』（）]+'
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub(' {2,}', ' ', text)  # 去除多余空格
    text = re.sub(r1, ' ', text)  # 去除多余空格
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('\u3000', '')
    text = text.replace('\u200b', '')
    text = text.replace('\\', '')

    # 将全角字符转换为半角字符
    text = re.sub('[\uFF00-\uFFEF]', lambda x: chr(ord(x.group(0)) - 0xFF00 + 0x20), text)
    return text.strip()

def extract_sentences(text):
    # 使用正则表达式将文本按句子分割
    sentences = re.split('[。！？]', text)
    return [clean_text(s) for s in sentences if len(clean_text(s)) > 0]

def segment_sentences(sentences):
    segmented_sentences = []
    for sentence in sentences:
        words = jieba.cut(sentence, cut_all=False)
        segmented_sentence = [word for word in words]
        segmented_sentences.append(segmented_sentence)
    return segmented_sentences


with open('dataFile/dataFilterData.txt', 'a', encoding='utf-8') as w:
    # 读取文件内容
    with open('dataFile/data.txt', 'r', encoding='utf-8') as r:
        content = r.read()

        # 清洗文本内容
        text = clean_text(content)
        # print(text)
        # 提取句子
        sentences = extract_sentences(text)
        pprint(sentences)
        print(len(sentences))
        for i in sentences:
            w.write(i + "\n")


        w.close()

