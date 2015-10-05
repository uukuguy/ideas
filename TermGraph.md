# 使用关联词向量空间模型实现语料库主题概览视图

## 背景
文本分析处理的对象是包含大量文本的语料库，无法人工一一翻阅的情况下，通过机器学习手段自动构建能表达语料库内容的主题概览视图，需要建立文本的表示模型。本研究从文本关键词共现关联的角度，探索通过关联词向量空间模型(CTVSM - Co-occurrence Term Vector Space Model)表达语料库的主题聚集、热点事件发现等。

## 算法设计

### 文本模型
词(Term)是文本中包含语义的最小组成单元，文本向量由文本中的词构成。不同词性的词在语义表达上的作用是不同的，虚词（副词、介词、连词等）是没有完整意义的词汇，只有语法意义或功能意义，不作为单独的语法成分。实词有实在意义，能够单独充当句子成分，一般能单独回答问题，包括名词、动词、形容词、数词、量词、代词六类。因此，本研究中文档向量由文档中的名词、动词、动名词构成。

\begin{equation}
D\_i = ( w\_{i,1}, w\_{i,2}, \ldots w\_{i,n} ) 
\end{equation}

上式中D$_i$是语料库中第_i_个文档向量，w$_{i,n}$是文档_i_中第_n_个关键词（仅选择名词、动词、动名词）。

\begin{equation}
Corpus = ( d\_1, d\_2, \ldots d\_n )
\end{equation}

以上是语料库模型，语料库Corpus由语料库中所有文档向量d$_n$组成。

### 词共现

#### 词的共现现象
通常，文本中出现的词汇都是为了帮助作者表达其主题思想，只不过每个词与主题的相关程度不同。词汇有其所属主题域，相同主题域中的词汇共同出现在同一篇文档中的概率相对较高，因此利用词的共现现象可以判断词汇与主题之间的相关程度。

假设文本空间D上有主题集合T和词汇集合W，其中，d$_i$ $\in$ D 代表第_i_篇文档，t$_k$ $\in$ T 代表文档空间中的第_k_个主题，S(t$_k$)为主题t$_k$的主题相关词集合，w$_m$ $\in$ W 为文档中出现的词汇。在主题t$_k$出现的情况下，词汇w$_m$出现的条件概率可表示为$P$($w_m$ $\mid$ $t_k$)。

> 共现率：词汇w$_i$与w$_j$的共现率是指这两个词在同一文本空间单位中共同出现的概率，即它们在文本空间中的联合概率，如下式所示：
\begin{equation}
P(w\_i, w\_j) = \sum_{i \in T}P(t) P(w\_i \mid t)P(w\_j \mid t), \forall w\_i, w\_j \in W
\end{equation}

设满足$P(\cdot) > \theta$的事件，即发生概率较高的事件，被称为显著性事件。其中$\theta$为显著性判别标准，通常与语料库的规模以及主题在文本空间中的分布有关。

当$P(w_m \mid t_k) \geq \theta$，即是显著性事件时，$w_m$是$t_k$的主题相关词汇，称为$w_m$与$t_k$主题相关。

当$P(w_m \mid t_k) < \theta$，即是非显著事件时，$w_m$不是$t_k$的主题相关词汇，称$w_m$与$t_k$主题无关。

推论：**如果两个词汇的共现为显著事件，则这两个词汇与某个共同的主题相关**。即：当$P(w_i, w_j) \geq 0$时，$\exists t_k$ 使得 $P(w_i \mid t_k) \geq 0$ 。

由此推论可知，如果两个词汇的共现率超过一定的阈值，则预示着这两个词很大可能是主题相关的。

#### 词的共现率计算

\begin{equation}
P(w\_i, w\_j) = \frac{\parallel Segmgnet(w\_i, w\_j) \parallel}{\parallel Segment \parallel}
\end{equation}

式中，$Segment(w_i, w_j)$表示文本空间中同时包含$w_i$和$w_j$的窗口单元集合，$Segment$表示文本空间中的窗口单元集合，$\parallel \cdot \parallel$表示集合中元素的个数。例如，将窗口单元设为一个自然段，则自然段内出现的词汇对视为它们的一次共现。

#### 共现词组合的抽取
共现词组合的抽取过程类似于利用关联规则的数据挖掘方法发现事务中频繁项集的过程。
支持度公式：

\begin{equation}
Support(w\_i, w\_j) = p(w\_i, w\_j)
\end{equation}

置信度公式：

\begin{equation}
Identify(w\_i, w\_j) = \frac{1}{2}\left( \frac{P(w\_i, w\_j)}{P(w\_i)} + \frac{P(w\_i, w\_j)}{P(w\_j)} \right)
\end{equation}

给定关联规则挖掘空间$S = (T, I, R, \theta)$，其中含义如下：
1. $T=\{t_1, t_2, \ldots, t_n\}$为S上的事务集合（Transaction Set），其中$t_n \in T$为S上的事务，即文本中的句子或自然段；
2. $I=\{ i_1, i_2, \ldots, i_m \}$为$S$上的项集（Item Set），其中$i_m \in I$为$S$上的项，即文本中的候选词汇；
3. $R=\{ r_1, r_2, \ldots, r_k \}$为$T$中蕴含的规则，其中$r_k=(i_x, i_y), i_x, i_y \in I$，即抽取出的共现词组合；
4. $\theta=\{ \alpha, \beta \}$为$S$上的约束，$\alpha$和$\beta$分别为给定的支持度与置信度的阈值。
则在空间$S$上的“事务-项”矩阵可以表示为$m \times n$的矩阵。其中行向量代表一个事务$t$（句子或自然段），列向量代表项$i$（候选词汇），矩阵为0-1矩阵。
空间$S$上的支持度公式：

\begin{equation}
support(i\_x, i\_y) = \sum_{k=1}^{n}\left ( i\_{kx}, i\_{ky} \right )
\end{equation}

空间$S$上的置信度公式：
\begin{equation}
identify(i\_x, i\_y) = \frac{1}{2}\left [ 
\frac{\sum\_{k=1}^{n}i\_{kx} \times i\_{ky}}{\sum i\_x} +
\frac{\sum\_{k=1}^{n}i\_{kx} \times i\_{ky}}{\sum i\_y} 
\right ]
\end{equation}

基于关联规则挖掘空间$S$的挖掘算法：
> 关联规则挖掘算法
> 输入：事务数据库
> 输出：规则集
> 过程：
> 1. 扫描事务数据库，构造$m \times n$的“事务-项”矩阵；
> 2. 生成$I$上的所有二阶的项组合$(i_x, i_y), \forall i_x, i_y \in I$；
> 3. 在项组合集上循环，利用上述$S$上的支持度和置信度公式计算项目组合$(i_x, i_y)$的支持度和置信度。如果支持度和置信度大于设定阈值，则将该组合加入$R$；
> 4. 返回规则集$R$，算法结束。

### 关联词向量空间模型（CTVSM）
与VSM的建模思想类似，CTVSM将文本表示为一个共现词组合的向量。设文档空间$D = \{d_1,d_2, \ldots,d_n\}$中包含$n$篇文档,在$D$上抽取出的共现词组合的集合为$C=\{c_1,c_2, \ldots,c_m \}$,其中$c_m$ 为抽取出的第$m$个共现词组合。则文档空间$D$可表示为一个$m \times n$的矩阵。其中行向量$d_i=(c_{i1},c_{i2}, \ldots,c_{im} )$代表一篇文档，列向量$c_j = (c_{1j}, c_{2j}, \ldots,c_{nj})$代表一个共现词汇组合在各文档中 的分布情况。矩阵中的元素$c_{ij}$表示文档$d_i$中共现词汇组合$c_j$的分布情况。如果共现词汇组合出现则相应的权值为1，如果不出现，则相应权值为0。
\begin{equation}
D = (d\_1, d\_2, \ldots, d\_n)^T
\end{equation}

\begin{equation}
d\_i = (c\_{i1}, c\_{i2}, \ldots, c\_{im})
\end{equation}

\begin{equation}
c_{ij} = 
\begin{cases} 
    0 & c\_j \notin d\_i  \\\\
    1 & c\_j \in d\_i
\end{cases}
\end{equation}


### 事件模型
事件e被定义为6元组格式：

\begin{equation}
e = (A, O, T, V, P, L)
\end{equation}

A是事件中的动作，O是事件中的对象，T是事件发生的时间，V是事件发生的环境（包括自然环境和社会环境），P是事件中动作执行过程的断言，L是语言表达式。

事件触发词(Event Trigger Word)，统计上表明词性主要是名词、动词、动名词。事件触发词窗口采用前3后2共6个词，第4个是事件触发词。

[事件本体以及突发事件语料库--CEC(Chinese Emergency Corpus)](http://blog.csdn.net/shijiebei2009/article/details/44538257)

[中文突发事件语料库](https://github.com/shijiebei2009/CEC-Corpus)

[中文环境突发事件语料库](https://github.com/shijiebei2009/CEEC-Corpus)

[基于CEC语料库挖掘要素识别规则，对新闻报道类生语料进行自动标注](https://github.com/shijiebei2009/CEC-Automatic-Annotation)

## 实验

## 结论

## 相关工作

### 词法分析

本研究选用的是开源的Jieba分词。目前主流分词工具通常包含了词性标注功能。Jieba分词完成词性标注的过程如下：

```
>>> import jieba.posseg as pseg
>>> words = pseg.cut(u'我爱北京天安门')
>>> for w in words:
...   print w.word, w.flag
...
我 r
爱 v
北京 ns
天安门 ns
```

### 关键词抽取
