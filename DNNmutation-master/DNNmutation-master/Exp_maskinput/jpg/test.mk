
我们通过将测试集中的图片按照类别求均值，然后选择阀值0.4，将求得的均值二值化，得到下面的十张图片。

其中黑色部分为0，白色部分为1。通过对比发现，上述图片和前面求得的图片具有一定的相似性。从逻辑上来说也是这样的，因为

只有在非零部分进行屏蔽才能导致网络的准确率发生比较大的变化。为了更科学的表单两者之间的关系，我们求出对应图片直接的交集所占的比例。

结果如下：

0.53，0.25，0.43，0.48，0.47，0.58，0.44，0.34，0.38，0.59


In test set, through computing the average of pictures of all ten categories and setting a threshold of 0.4, we binarized these ten averages and got pictures as below:

In these pictures, black parts stand for 0 while white for 1. We found some similarity between these pictures and pictures computed before, which can also be explained logistically. This is because the network’s accuracy will only be changed greatly when no-zero parts are blocked. To represent this relationship more quantitatively, we calculated the ratios of intersection in every pair. Results are as below:
0.53, 0.25, 0.43, 0.48, 0.47, 0.58, 0.44, 0.34, 0.38, 0.59
