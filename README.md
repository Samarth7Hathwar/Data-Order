# Data-Order


Description :
Typically, during training, we present samples to our neural network model in random order. From prior work in the field of continual learning [1], we know that this is not necessarily the best approach. When gradually
increasing the size of the dataset, at least for CIFAR100, it appears that adding the most dissimilar class(es) next consistently outperforms other approaches. 

This topic aims to first replicate these results and expand them to large-scale with ImageNet-1k, with the ultimate goal of faster training through fewer images processed. From there, students will further investigate whether the graph structure of classes in ImageNet, which was derived from WordNet, can possibly yield
similar results.

[1] Mundt, Martin, et al. "A Wholistic View of Continual Learning with Deep Neural
Networks: Forgotten Lessons and the Bridge to Active and Open World Learning." arXiv
preprint arXiv:2009.01797 (2020). https://arxiv.org/abs/2009.01797
