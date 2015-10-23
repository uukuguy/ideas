# 使用关联词向量空间模型实现语料库主题概览视图

## 背景
文本分析处理的对象是包含大量文本的语料库，无法人工一一翻阅的情况下，通过机器学习手段自动构建能表达语料库内容的主题概览视图，需要建立文本的表示模型。本研究从文本关键词共现关联的角度，探索通过关联词向量空间模型(CTVSM - Co-occurrence Term Vector Space Model)表达语料库的主题聚集、热点事件发现等。


## 关联词向量空间模型

### 向量空间模型（VSM）
词(Term)是文本中包含语义的最小组成单元，文本向量由文本中的词构成。不同词性的词在语义表达上的作用是不同的，虚词（副词、介词、连词等）是没有完整意义的词汇，只有语法意义或功能意义，不作为单独的语法成分。实词有实在意义，能够单独充当句子成分，一般能单独回答问题，包括名词、动词、形容词、数词、量词、代词六类。因此，本研究中文本关键词的选择由文本中的名词、动词、动名词构成。\citet{kottwitz2011latex}提出了一种算法，\citet{NI-OVERSAMPLING}提出了另一种算法。

\begin{equation}
D\_i = ( w\_{i,1}, w\_{i,2}, \ldots w\_{i,n} )
\end{equation}

上式是原生文本向量，即常规的向量空间模型，其中$D_i$是语料库中第$i$个文本向量，$w_{i,n}$是文本$D_i$中第$n$个关键词（仅选择名词、动词、动名词）。

\begin{equation}
Corpus = \{ d\_1, d\_2, \ldots d\_n \}
\end{equation}

以上是语料库模型，语料库Corpus由语料库中所有文本向量d$_n$的集合。

### 词的共现现象
通常，文本中出现的词汇都是为了帮助作者表达其主题思想，只不过每个词与主题的相关程度不同。词汇有其所属主题域，相同主题域中的词汇共同出现在同一篇文档中的概率相对较高，因此利用词的共现现象可以判断词汇与主题之间的相关程度。

假设文本空间$D$上有主题集合$T$和词汇集合$W$，其中，d$_i$ $\in D$代表第$i$篇文档，t$_k$ $\in T$代表文档空间中的第$k$个主题，$S(t_k)$为主题$t_k$的主题相关词集合，$w_m \in W$为文档中出现的词汇。在主题$t_k$出现的情况下，词汇$w_m$出现的条件概率可表示为$P(w_m \mid t_k)$。

\begin{definition}[共现率]
词汇$w_i$与$w_j$的共现率是指这两个词在同一文本空间单位中共同出现的概率，即它们在文本空间中的联合概率， \label{def1}如下式所示：
\begin{equation}
P(w\_i, w\_j) = \sum_{i \in T}P(t) P(w\_i \mid t)P(w\_j \mid t), \forall w\_i, w\_j \in W
\label{formula1}
\end{equation} 
\end{definition}


\begin{definition}[显著性事件]
设满足$P(\cdot) > \theta$的事件，即发生概率较高的事件，被称为显著性事件。其中$\theta$为显著性判别标准，通常与语料库的规模以及主题在文本空间中的分布有关。 \label{def2}
\end{definition}

\begin{definition}[主题相关]
当$P(w_m \mid t_k) \geq \theta$，即是显著性事件时，$w_m$是$t_k$的主题相关词汇，称为$w_m$与$t_k$主题相关。
\end{definition} \label{def3}

\begin{definition}[主题无关]
当$P(w_m \mid t_k) < \theta$，即是非显著事件时，$w_m$不是$t_k$的主题相关词汇，称$w_m$与$t_k$主题无关。 \label{def4}
\end{definition}

\begin{corollary}
如果两个词汇的共现为显著事件，则这两个词汇与某个共同的主题相关。即：当$P(w_i, w_j) \geq \theta$时，$\exists t_k$ 使得 $P(w_i \mid t_k) \geq \theta$ 且 $P(w_j \mid t_k) \geq \theta$。 \label{coro1}
\end{corollary}

由此推论 \ref{coro1}可知，如果两个词汇的共现率超过一定的阈值，则预示着这两个词较大可能是主题相关的。

### 词的共现率计算
由定义 \ref{def1}可知，词的共现率是两个词在同一主题内同时出现的概率之和，然而在实际文本建模过程中， 主题是文本的隐含变量，无法准确获得。因此，通过统计两个词在文本中共现的次数计算它们的共现率。假设一个窗口单元（例如，一句话、一个段落）代表一个主题，文本空间中有$n$个窗口单元，则主题的先验概率为$P(t) = 1 / n$。当词$w$在窗口单元出现，则其后验概率为$P(w \mid t) = 1$。根据公式
\ref{formula1}，如果词$w_i$和$w_j$共同出现在$x$个不同的窗口单元中，它们的联合概率为$P(w_i, w_j)= x / n$。因此词的共现率可以由公式 \ref{formula2}计算得出。

\begin{equation}
P(w\_i, w\_j) = \frac{\parallel Segmgnet(w\_i, w\_j) \parallel}{\parallel Segment \parallel}
\label{formula2}
\end{equation} 

式中，$Segment(w_i, w_j)$表示文本空间中同时包含$w_i$和$w_j$的窗口单元集合，$Segment$表示文本空间中的窗口单元集合，$\parallel \cdot \parallel$表示集合中元素的个数。例如，将窗口单元设为一个自然段，则自然段内出现的词汇对视为它们的一次共现。

### 共现词组合的抽取
共现词组合的抽取过程类似于利用关联规则的数据挖掘方法发现事务中频繁项集的过程。通过关联规则挖掘算法，挖掘语料库中词与词之间的关系 ，其中支持度由公式 \ref{formula_support0}表示，置信度由公式 \ref{formula_identify0}表示。

\begin{equation}
Support(w\_i, w\_j) = P(w\_i, w\_j)
\label{formula_support0}
\end{equation}

\begin{equation}
Identify(w\_i, w\_j) = \frac{1}{2}\left( \frac{P(w\_i, w\_j)}{P(w\_i)} + \frac{P(w\_i, w\_j)}{P(w\_j)} \right)
\label{formula_identify0}
\end{equation}

给定关联规则挖掘空间$S = (T, W, R, \theta)$，其中各项含义如下：

1. $T=\{t_1, t_2, \ldots, t_m\}$为空间$S$上的文本段集合（Text Segment Set），其中$t_m \in T$为空间$S$上的文本段，即文本中的句子、自然段或全文；
2. $W=\{ w_1, w_2, \ldots, w_n \}$为空间$S$上的词汇集（Word Set），其中$w_n \in W$为空间$S$上的候选词汇；
3. $R=\{ r_1, r_2, \ldots, r_k \}$为文本段集合$T$中蕴含的规则，其中$r_k=(w_x, w_y), w_x, w_y \in W$，即抽取出的共现词组合；
4. $\theta=\{ \alpha, \beta \}$为空间$S$上的约束，$\alpha$和$\beta$分别为给定的支持度与置信度的阈值。

在空间$S$上的“文本段-词汇”矩阵可以表示为$m \times n$的矩阵。其中行向量代表一个文本段$t$（句子、自然段或全文），列向量代表词汇$w$（候选词汇），矩阵为0-1矩阵。

空间$S$上的支持度由公式 $\ref{formula_support}$表示，置信度由公式 $\ref{formula_identify}$表示。

\begin{equation}
support(w\_x, w\_y) = \sum_{k=1}^{m} (w\_{kx} \times w\_{ky}) 
\label{formula_support} 
\end{equation}

\begin{equation}
identify(w\_x, w\_y) = \frac{1}{2}\left [ 
\frac{\sum\_{k=1}^{m}w\_{kx} \times w\_{ky}}{\sum w\_x} +
\frac{\sum\_{k=1}^{m}w\_{kx} \times w\_{ky}}{\sum w\_y} 
\right ]
\label{formula_identify}
\end{equation}

在上述定义的关联规则挖掘空间$S$上，有以下挖掘算法：

\begin{algorithm}[htb]
\caption{利用关联规则挖掘共现词组合算法}
\label{alg:co_word}
\begin{algorithmic}[1]
\REQUIRE 文本语料库$Corpus$，阈值$\theta$
\ENSURE 共现词组合列表$R$
\STATE 读取语料库$Corpus$，分词后生成$m \times n$的“文本段-词汇”矩阵$\{ P, W\}$；
\STATE 生成所有二阶词对组合$(w_x, w_y), \forall w_x, w_y \in W$；
\FOR{ each $(w_x, w_y)$}
    \STATE // 按公式 \ref{formula_support}计算支持度$s$
    \STATE $support(w_x, w_y) \to s$
    \STATE // 按公式 \ref{formula_identify}计算置信度$i$ 
    \STATE $identify(w_x, w_y) \to i$     
    \STATE // 如果$(s,i)$大于设定阈值$\theta$，则将该组合加入$R$
    \IF {$s \geq \alpha \wedge i \geq \beta$}
        \STATE $(w_x, w_y) \to R$
    \ENDIF
\ENDFOR
\RETURN $R$
\end{algorithmic}
\end{algorithm}

此算法只在开始时读取一次语料库，在内存中构建矩阵，之后利用矩阵运算求解词共现的支持度和置信度。运行效率较高，且算法的复杂度为$O(n^2)$。

### 关联词向量空间模型（CTVSM）
与VSM的建模思想类似，CTVSM将文本表示为一个共现词组合的向量。设文档空间$D = \{d_1,d_2, \ldots,d_n\}$中包含$n$篇文档,在$D$上抽取出的共现词组合的集合为$C=\{c_1,c_2, \ldots,c_m \}$,其中$c_m$ 为抽取出的第$m$个共现词组合。则文档空间$D$可表示为一个$m \times n$的矩阵。

\begin{equation}
D = (d\_1, d\_2, \ldots, d\_n)^T
\end{equation}

其中行向量$d_i=(c_{i1},c_{i2}, \ldots,c_{im} )$代表一篇文档，


\begin{equation}
d\_i = (c\_{i1}, c\_{i2}, \ldots, c\_{im})
\end{equation}

列向量$c_j = (c_{1j}, c_{2j}, \ldots,c_{nj})$代表一个共现词汇组合在各文档中 的分布情况。矩阵中的元素$c_{ij}$表示文档$d_i$中共现词汇组合$c_j$的分布情况。如果共现词汇组合出现则相应的权值为1，如果不出现，则相应权值为0。

\begin{equation}
c_{ij} = 
\begin{cases} 
    0 & c\_j \notin d\_i  \\\\
    1 & c\_j \in d\_i
\end{cases}
\end{equation}


## 事件模型
事件e被定义为6元组格式：

\begin{equation}
e = (A, O, T, V, P, L)
\end{equation}

A是事件中的动作，O是事件中的对象，T是事件发生的时间，V是事件发生的环境（包括自然环境和社会环境），P是事件中动作执行过程的断言，L是语言表达式。

事件触发词(Event Trigger Word)，统计上表明词性主要是名词、动词、动名词。事件触发词窗口采用前3后2共6个词，第4个是事件触发词。

* [事件本体以及突发事件语料库--CEC(Chinese Emergency Corpus)][1]
* [中文突发事件语料库][2]
* [中文环境突发事件语料库][3]
* [基于CEC语料库挖掘要素识别规则，对新闻报道类生语料进行自动标注][4]

[1]: http://blog.csdn.net/shijiebei2009/article/details/44538257
[2]: https://github.com/shijiebei2009/CEC-Corpus
[3]: https://github.com/shijiebei2009/CEEC-Corpus
[4]: https://github.com/shijiebei2009/CEC-Automatic-Annotation

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
词汇与主题相关
