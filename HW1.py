import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

#新增雲端讀取權限，自定義字體，用以顯示中文
#https://ithelp.ithome.com.tw/articles/10234373
#https://colab.research.google.com/github/willismax/matplotlib_show_chinese_in_colab/blob/master/matplotlib_show_chinese_in_colab.ipynb#scrollTo=E6BV3V81MXHe
import os
from google.colab import drive

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt 

drive.mount('/content/drive')
os.listdir()
!wget -O taipei_sans_tc_beta.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download
!mv taipei_sans_tc_beta.ttf /usr/local/lib/python3.6/dist-packages/matplotlib//mpl-data/fonts/ttf
# 自定義字體變數
myfont = FontProperties(fname=r'/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/taipei_sans_tc_beta.ttf')


def build_word_vector(text):
    word2id = {w: i for i, w in enumerate(sorted(list(set(reduce(lambda a, b: a + b, text)))))}
    id2word = {x[1]: x[0] for x in word2id.items()}
    wvectors = np.zeros((len(word2id), len(word2id)))
    for sentence in text:
        for word1, word2 in zip(sentence[:1], sentence[1:]):
            id1, id2 = word2id[word1], word2id[word2]
            wvectors[id1, id2] += 1
            wvectors[id2, id1] += 1
    return wvectors, word2id, id2word


def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.sum(np.power(v1, 2))) * np.sqrt(np.sum(np.power(v1, 2))))


def visualize(wvectors, id2word):
    np.random.seed(10)
    fig = plt.figure(dpi = 150)
    U, sigma, Vh = np.linalg.svd(wvectors)
    ax = fig.add_subplot(111)
    ax.axis([-1, 1, -1, 1])
    for i in id2word:
        ax.text(U[i, 0], U[i, 1], id2word[i], alpha=0.3, fontsize=20,fontproperties=myfont)
    plt.show()


if __name__ == '__main__':
    text = [["六十六","歲", "的", "陸老頭", ],
            ["蓋", "六十六","間", "樓", ],
            ["買", "六十六","簍", "油"],
            ["養", "六十六","頭", "牛", ],
            ["栽", "六十六","棵", "垂楊柳", ],
            ]

    wvectors, word2id, id2word = build_word_vector(text)

    print(word2id)

    print(id2word)


    visualize(wvectors, id2word)
