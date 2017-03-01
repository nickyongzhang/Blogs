MV-CNN for extractive document summarization
================

引子
----

本文简单介绍我发在IEEE Trans. on Cybernetics上的一篇论文，成果已不是最新，文章是2015年的工作，投稿周期较长导致最近才发表。此文不是推介，实乃当下正在找工作，需要对自己PhD期间做过的工作有个梳理。故，写与自己。

简介
----

就笔者所知，此文是第一次将multi-view learning和CNN结合到一起，新的model应用于抽取型文本摘要，在DUC的benchmark dataset上取得了state-of-art performance。我们工作的主要贡献如下：

-   首次利用multi-view learning的complementaty和concensus原则，提高CNN的学习能力，实验证明multi-view learning是CNN的一个可行方向。
-   利用pre-train word embedding matrix, 实现模型end-to-end可训练，不再需要feature engineering，模型易于使用。
-   首次提出sentence position embedding技术，提高文档摘要任务的准确率。

以下，笔者将从文档摘要，CNN，multi-view learning几个方面做简单介绍。

文档摘要
--------

信息爆炸时代如何快速提取到有用信息非常重要，文档摘要作为信息提取领域的重要技术，长期以来一直备受关注，近年来随着机器学习的火热，越来越多的人尝试用机器学习来解决这一问题。文档摘要主要有两个分支：extractive summarization和abstractive summarization。前者直接提取文档活文集里的句子组合成最终的摘要，后者还需要做语法检查，语义重组等复杂处理。后者提取到的摘要与人工摘要更加类似，但是技术性复杂，本文focus在extractive summarization。我们首先将文档或文集分割成句子，然后利用我们的模型对每个句子建模并打分，之后根据句子得分排序，取分数居前的句子组合成最终摘要。最后的摘要有字数上线或者句子数目上限。此处值得提到的是，对于文集摘要任务，来自不同文档的句子可能十分相同（任务是针对相同topic的文档提取摘要）而且同时得分很高，我们将剔除重复性句子。