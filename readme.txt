 三个工程均训练<mnist>数据集
-1. pytorch-artificial-train
     总结常用的写作手法,将流程规范化,本人也更看好这个模型

-2. pytorch-lighting-train
     使用 lighting框架, 让整体流程更简洁, 更像是一个玩具, 灵活性, 不清楚数据的整个流过程, 在训练的时候就出现 训练日志与验证日志交叉打印的现象,除了升级或更换版本,
     没有想到其他更好的办法.

-3. pytorch-normal-train
     最最常用(最最原始)的使用方法



a. 使用该工程的前提是先下载mnist数据集,并将其拆解成.png/.txt成对的存在.这样可以练习普通正常的数据加载过程
   python 下载minist数据集.py

