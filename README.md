1.ED网络和MUNET的输出通道可根据具体使用的数据集大小在main函数里面做出更改2.实现ED-MUNET时网络需要将ED网络的结果作为MUNET的输入，在此需要用Matlab的image函数将ED网络的输出转换为图像，因此需要将MUNET的输入输出数据改为图像格式，程序需要微调
