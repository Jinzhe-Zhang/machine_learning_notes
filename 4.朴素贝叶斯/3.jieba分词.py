import os
import jieba
content = '中文文本分类是文本处理中的一个基本问题。'
seg_list1=jieba.cut(content, cut_all=False)
print('默认切分模式：')
print (' '.join(seg_list1))

seg_list1=jieba.cut(content, cut_all=True)
print('全切分模式：')
print (' '.join(seg_list1))

seg_list1=jieba.cut_for_search(content)
print('搜索引擎分词模式：')
print (' '.join(seg_list1))