# 组织化导入库
import numpy as np
import gensim, jieba, gensim.corpora as corpora
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from gensim.models import CoherenceModel, LdaModel
import re
from collections import Counter

# 加载停用词
def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().split("\n") + ["\u3000"]

# 预处理文本，减少函数
def preprocess_text(text, stop_words):
    for phrase in ["本书来自www.cr173.com免费txt小说下载站", "更多更新免费电子书请关注www.cr173.com", "\u3000"]:
        text = text.replace(phrase, "")
    words = jieba.lcut(text)
    chars = [char for char in text]
    return [w for w in words if w not in stop_words and w != ' '], \
           [c for c in chars if c not in stop_words and c != ' ']

# 重构提取段落的函数，整合为一个通用函数
def extract_paragraphs(para_num, token_num, book_para_data, cut_option='word'):
    dataset, sampled_labels = [], []
    para_num_per_book = int(para_num / len(book_para_data)) + 1
    for label, paragraphs in book_para_data.items():
        label_paragraphs = sum((p[:token_num] for p in paragraphs if len(p) >= token_num), [])
        if len(label_paragraphs) < para_num_per_book:
            label_paragraphs *= int(para_num_per_book / len(label_paragraphs) + 1)
        sampled_paragraphs = np.random.choice(label_paragraphs, para_num_per_book, replace=False)
        dataset.extend(sampled_paragraphs)
        sampled_labels.extend([label] * para_num_per_book)

    return dataset[:para_num], sampled_labels[:para_num]

# LDA模型训练与主题提取整合
def train_and_get_document_topics(train_corpus, id2word, num_topics):
    lda_model = LdaModel(corpus=train_corpus, id2word=id2word, num_topics=num_topics,
                         random_state=100, update_every=1, chunksize=1000, passes=10,
                         alpha='auto', per_word_topics=True, dtype=np.float64)

    document_topics = []
    for item in train_corpus:
        tmp = {index: v for index, v in lda_model.get_document_topics(item)}
        document_topics.append([tmp.get(i, 0) for i in range(num_topics)])
    return lda_model, document_topics

# 主函数`
def main():
    # 省略了一些初始化和导入数据的代码
    # 重构核心段落调用`extract_paragraphs`进行测试

    # 示例调用提取功能和训练模型
    para_num = 20
    token_num = 100
    corpus = []
    with open("C:\\Users\\Acer\\Desktop\\Chinese corpus\\4.txt", "r", encoding="utf-8") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        corpus += text
    regex_str = ".*?([^\u4E00-\u9FA5]).*?"
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    symbol = []
    for j in range(len(corpus)):
        corpus[j] = re.sub(english, "", corpus[j])
        symbol += re.findall(regex_str, corpus[j])
    count_ = Counter(symbol)
    count_symbol = count_.most_common()
    noise_symbol = []
    for eve_tuple in count_symbol:
        if eve_tuple[1] < 200:
            noise_symbol.append(eve_tuple[0])
    noise_number = 0
    for line in corpus:
        for noise in noise_symbol:
            line.replace(noise, "")
            noise_number += 1
    # print(corpus)
    book_para_jieba = []
    result = []
    for para in corpus:
        word = []
        for j in para:
            word += j
        result += jieba.lcut(para)
        # token += word
    with open("C:\\Users\\Acer\\Desktop\\jyxstxtqj_downcc.com",
              encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()
        for i in con:
            i = i.replace("\n", "")  # 去掉读取每一行数据的\n
            stop_words.add(i)
    for word in result:
        if word not in stop_words:
            book_para_jieba.append(word)
    dataset, labels = extract_paragraphs(para_num, token_num, book_para_jieba, 'word')
    id2word = corpora.Dictionary(dataset)
    # 划分数据，训练LDA模型，评价模型等
    ...

if __name__ == '__main__':
    main()