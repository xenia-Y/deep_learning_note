# 深度学习自己的笔记
## 正则化
在原来loss function 上的基础中，加入正则化项（regularizer）来实现权重衰减（weight decay).
简单的说就是：我们修改学习算法，使其降低泛化误差而非训练误差
正则化项一般是模型复杂度的单调递增的函数，模型越复杂，正则化值越大。
### L1 范数与L2 范数
第一项是L2正则化，第二项是L1正则化
$$
 \begin{aligned} J(\vec{w}, b)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right) &+\frac{\lambda}{2 m}\|\vec{w}\|_{2}^{2} \\ &+\frac{\lambda}{m}\|\vec{w}\|_{1} \end{aligned}  
 $$

L1范数可以使weight 稀疏： 任何的正则化算子，如果他在Wi=0的地方不可微，并且可以分解为“求和” 的形式，那么这个正则化算子就可以实现稀疏。

不是特别理解为什么L2范数只能使weight接近于0，不能等于0.
L1 sparse
L2 uniform

（Regularisation is doing in training and for L1/L2 it change the loss and for the dropout, it introduce $p_{d} \neq 0$
For validation, L1/L2 no optimized, and for dropout, need to cancel dropout.）
也就是这些优化只在training的时候用
dropout 设置为0的参数是随机选择，实现了一个uniform的效果，所以更像L2
## dropout (不是特别理解）
老师上课说drpout similar to L2 regulazation，有点不理解，L2不会设置weight为0，L1会使weight=0，为什么dropout会和L2 regulazation更像呢？
#### 有用的链接
1. L2正则化：[https://blog.csdn.net/u010725283/article/details/79212762]()
2. L1/L2正则化： [https://satishkumarmoparthi.medium.com/why-l1-norm-creates-sparsity-compared-with-l2-norm-3c6fa9c607f4]()
