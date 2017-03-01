MV-CNN for extractive document summarization
================

引子
----

本文简单介绍我发在IEEE Trans. on Cybernetics上的一篇论文，成果已不是最新，文章是2015年的工作，投稿周期较长导致最近才发表。此文不是推介，实乃当下正在找工作，需要对自己PhD期间做过的工作有个梳理。故，写与自己。

简介
----

就笔者所知，此文是第一次将multi-view learning和CNN结合到一起，新的model应用于抽取型文本摘要，在DUC的benchmark dataset上取得了state-of-art performance。在DUC2006数据集上，与literature里效果最好的方法相比，我们的model将Rouge-1, Rouge-2, 和Rouge-SU4三个metrics上的表现提高了1.28%，4.35%，和3.68%。我们工作的主要贡献如下：

-   首次利用multi-view learning的complementaty和concensus原则，提高CNN的学习能力，实验证明multi-view learning是CNN的一个可行方向。
-   利用pre-train word embedding matrix, 实现模型end-to-end可训练，不再需要feature engineering，模型易于使用。
-   首次提出sentence position embedding技术，提高文档摘要任务的准确率。

以下，笔者将从文档摘要，CNN，word embedding, multi-view learning几个方面做简单介绍。然后介绍文章提出的模型。

文档摘要
--------

信息爆炸时代如何快速提取到有用信息非常重要，文档摘要作为信息提取领域的重要技术，长期以来一直备受关注，近年来随着机器学习的火热，越来越多的人尝试用机器学习来解决这一问题。文档摘要主要有两个分支：extractive summarization和abstractive summarization。前者直接提取文档活文集里的句子组合成最终的摘要，后者还需要做语法检查，语义重组等复杂处理。后者提取到的摘要与人工摘要更加类似，但是技术性复杂，本文focus在extractive summarization。我们首先将文档或文集分割成句子，然后利用我们的模型对每个句子建模并打分，之后根据句子得分排序，取分数居前的句子组合成最终摘要。最后的摘要有字数上线或者句子数目上限。此处值得提到的是，对于文集摘要任务，来自不同文档的句子可能十分相同（任务是针对相同topic的文档提取摘要）而且同时得分很高，我们将剔除重复性句子。完整的流程如下图： ![framework](https://raw.githubusercontent.com/nickzylove/Blogs/master/MV-CNN_files/figure-markdown_github/framework.png)

CNN
---

Convolutional Neural Network (CNN)可以说是近几年最火的算法之一了，凡做图像必用CNN，因为其良好的local representation的能力可以有效提取到图像的局部特征。最近CNN也被广泛应用到NLP领域，本证明学习能力依然出众。基本的CNN模型可以参见Stanford CS231n课程[CNN for visual recogonition](http://cs231n.github.io/convolutional-networks/)。本文使用的基本CNN结构包括两层convolution layer和一层max pooling layer。选择两层convolution layer的原因是相较于一层学习能力有较大幅度提升，而再增加层数效果提升并不明显却导致复杂度急剧上升。

![framework](https://raw.githubusercontent.com/nickzylove/Blogs/master/MV-CNN_files/figure-markdown_github/cnn_structure.png)

Word Embedding
--------------

词向量近年几乎已经为所有NLP任务使用，其将每个单词的向量表示映射到一个相对低维的空间，并且保存了词与词之间的语义关联: <!-- ![framework](https://raw.githubusercontent.com/nickzylove/Blogs/master/MV-CNN_files/figure-markdown_github/word_embedding.png) --> 本文使用了pre-trained word2vec matrix作为词向量的初始化，然后在训练模型的时候通过SGD对词向量做更新，从而得到特定任务下的词向量，实验证明,利用在大语料Google News上训练的word embedding matrix做初始化可以显著提升预测能力。

Multi-view Learning
-------------------

> Multiview learning is a paradigm designed for problems, where data come from diverse sources or feature subsets, also known as, multiple views. It employs one function to model a particular view and jointly optimizes all the functions to exploit the redundant views of the same input data and improve the learning performance \[1\].

Multi-view Learning从多角度去提取相同输入数据的信息，这一点人工摘要提取十分类似。人工摘要提取的时候，会让多个summarizer对同一个文档或文集做出总结，每个summarizer的认识角度都是不同的，提取出来的最终结果也会不尽相同。我们利用multi-view learning来模仿多个summarizer，利用complementary and consensus原则使文本信息得到最有效的提取和表示。

Multi-view learning的主要技术包括Co-training，Multiple kernel learning algorithms，和Subspace learning-based approaches。本文中使用的multi-view leanring类似于Multiple kernel learning，同时兼顾co-training的属性。

Mutliview lerning成功的关键在于两条原则，即complementary and consensus principles. 前者旨在利用数据隐含的互补的信息，后者则目的在于最大化不同views上训练得到的learners的一致性。

Methodology
-----------

我们的新模型成功将multi-view learning引入经典CNN模型，利用two-level的complementary principle和consensus principle提高了经典CNN模型的学习能力。我们的模型的整体架构如下图： <!-- ![framework](https://raw.githubusercontent.com/nickzylove/Blogs/master/MV-CNN_files/figure-markdown_github/framework.png) --> 多个CNN被用于提取multiple views, 我们将每个CNN提取到的句子向量表示结合起来作为句子的最终表示，这个是intermediate stage的complementary principle。另一方面，我们也让每个CNN直接输出预测结果，然后将预测结果结合起来作为final stage的complementary principle。各个CNN 在讲如何引入multi-view learning之前，我先介绍一下我们在文章中另一创新点sentence position embedding。

### sentence position embedding

对于一个文档，句子所处的位置对于信息理解非常重要。例如，很多情况下，作者喜欢把重要信息或者概括性信息放在文章的开头或者结尾。在文档摘要的相关文献当中就有一个利用句子位置信息的方法，叫做LEAD。它使用每个文章的开头几个句子作为文章的摘要，也取得不错效果。为了利用好句子的位置信息，我们引入了sentence position embedding。为什么要引入sentence position embedding？

句子的位置信息可以用简单的自然数来表示，我们在意的是句子所在大概位置，所以可以按照下面的公式来表示句子位置信息。*S*<sub>1 : 3</sub>表示文章中的前三句，*S*<sub>−3 : −1</sub>表示后三句，其他位置的句子统一用2来表示。这样的表示方法比较粗糙，但是也非常的简单，我们模型的最重要的一个目的就在于减少人工的feature engineering，所以我们使用了最简单而且直观的方法。

<!-- ![framework](https://raw.githubusercontent.com/nickzylove/Blogs/master/MV-CNN_files/figure-markdown_github/sent_pos_embedding_equ.png) -->
句子位置信息可以与CNN提取到的句子语义信息结合，一起作为顶层分类器（这里就是softmax classifier)的输入。CNN提取到的句子语义信息以vector的形式表现，在我们的model里这个vector的长度为300（后文会有更多解释）。如果我们只是用自然数和这个语义向量连接，句子位置信息很有可能被语义向量所淹没，因此我们借助embedding的思想将自然数映射到高纬度的空间。事实上，word embedding本身也是讲interger表示映射为dense vector表示的过程。此处，我们的sentence position embedding的维度定为了100，并且初始化为随机的，它也会在训练model的过程中被更新。

Reference
---------

\[1\] C. Xu, D. Tao, and C. Xu, “A survey on multi-view learning,” Neural Comput. Appl., vol. 23, nos. 7–8, pp. 2031–2038, 2013.
