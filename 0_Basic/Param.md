## TensorFlow 参数

1. batchsize, iteration, epoch的差别
batchsize：批大小。在深度学习中，一般采用SGD训练，
即每次训练在训练集中取batchsize个样本训练；一次Forword运算以及BP运算中所需要的训练样本数目，其实深度学习每一次参数的更新所需要损失函数并不是由一个{data：label}获得的，而是由一组数据加权得到的，这一组数据的数量就是[batch size]。当然batch size 越大，所需的内存就越大，要量力而行
iteration：1个iteration等于使用batchsize个样本训练一次；
epoch：1个epoch等于使用训练集中的全部样本训练一次；完成一次forword和一次BP运算
one epoch = numbers of iterations = N = 训练样本的数量/batch size
一次epoch 总处理数量 = iterations次数 * batch_size大小

