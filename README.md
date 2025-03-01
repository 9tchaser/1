# 使用条件生成式对抗网络（CGAN）模型补充稀疏医学图像数据

**需要各位完成的**
- 学习图像的读取、处理等方法，可参考<https://github.com/eastmountyxz/ImageProcessing-Python.git>
- 学习神经网络尤其是CGAN的概念，可参考吴恩达和李沐的课程，github上也可以搜出来很多如<https://github.com/zergtant/pytorch-handbook.git>
- 学习一些包和调参技巧，当一个合格的调包虾+调参侠（X）
- 有问题随时交流，互相学习！

## 数据来源：超声CT检测的乳腺癌影像
- 老师还没给

## 需要完成以下目的：
1. 对乳腺癌患者的USCT图像数据进行预处理和标签等操作
2. 设计CGAN模型，该模型包括生成器和判别器。生成器的输入应该是随机噪声和条件信息（例如癌变类型），输出是一张合成的乳腺癌变区图像。判别器的任务是区分生成的图像和真实的图像。
3. 训练模型
4. 生成新图像，评估，整合

## 目前完成的工作：
- 由于没有USCT的数据，我从网上找了些超声数据，见data文件夹，权当练习
- 图像的分类，标签，预处理……代码见code文件夹
- 大致的网络框架，见code文件夹。一个是用FashinMINST数据集跑的，效果一般般。另一个是自己找的超声数据，没跑通，还在试，而且数据量比较少。
