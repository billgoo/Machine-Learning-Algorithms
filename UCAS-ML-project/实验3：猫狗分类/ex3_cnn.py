
# coding: utf-8

# ![image.png](attachment:image.png)

# 我们使用CIFAR10数据集。CIFAR10数据集包含60,000张32x32的彩色图片，10个类别，每个类包含6,000张。其中50,000张图片作为训练集，10000张作为验证集。这次我们只对其中的猫和狗两类进行预测。

# ![%E6%95%B4%E4%B8%AA%E6%B5%81%E7%A8%8B.png](attachment:%E6%95%B4%E4%B8%AA%E6%B5%81%E7%A8%8B.png)

# 导入需要的包，os:提供了丰富的方法来处理文件和目录

# In[1]:


#导入需要的包
import paddle as paddle
import paddle.fluid as fluid
import os


# ![%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE.png](attachment:%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE.png)

# 1构造读取CIFAR10数据集的train_reader和test_reader，指定一个Batch的大小为128，也就是一次训练或验证128张图像。 2.paddle.dataset.cifar.train10()或test10()接口已经为我们对图片数组转换的处理。

# 由于本次实践的数据集稍微比较大，以防出现不好下载的问题，为了提高效率，可以用下面的代码进行数据集的下载。

# In[2]:


# !mkdir -p  /home/aistudio/.cache/paddle/dataset/cifar
# wget将下载的文件存放到指定的文件夹下，同时重命名下载的文件，利用-O
#!wget "http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz
#!mv cifar-10-python.tar.gz  /home/aistudio/.cache/paddle/dataset/cifar/


# In[3]:


BATCH_SIZE = 128 # 每次取数据的个数
#将训练数据和测试数据读入内存
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(), #获取cifa10训练数据
                          buf_size=128 * 100),                  #在buf_size的空间内进行乱序
    batch_size=BATCH_SIZE)                                #batch_size:每个批次读入的训练数据
test_reader = paddle.batch(
    paddle.dataset.cifar.test10(),                        #获取cifa10测试数据     
    batch_size=BATCH_SIZE)                                #batch_size:每个批次读入的测试数据


# 尝试打印一下，观察一下CIFAR10数据集

# In[4]:


temp_reader = paddle.batch(paddle.dataset.cifar.train10(),
                            batch_size=3)
temp_data=next(temp_reader())
print('temp_data_shape',type(temp_data))
print('temp_data_shape：',len(temp_data))
print(temp_data)


# ![%E9%85%8D%E7%BD%AE%E7%BD%91%E7%BB%9C.png](attachment:%E9%85%8D%E7%BD%AE%E7%BD%91%E7%BB%9C.png)

# 配置网络主要是用来生组建一个Program，包括三个部分：1.网络模型2.损失函数3.优化函数

# image 和 label 是通过 fluid.layers.data 创建的两个输入数据层。其中 image 是 [3, 32, 32] 维度的浮点数据; label 是 [1] 维度的整数数据。 

# 这里需要注意的是:
# Fluid中默认使用 -1 表示 batch size 维度，默认情况下会在 shape 的第一个维度添加 -1 。 所以 上段代码中， 我们可以接受将一个 [128, 3, 32, 32] 的numpy array传给 image 。Fluid中用来做类别标签的数据类型是 int64，并且标签从0开始。

# In[5]:


#定义输入数据
data_shape = [3, 32, 32]#3表图像RGB三通道，32*32的彩色图片
# 定义全局变量image和label
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
print('images_shape:',images.shape)


# 在CNN模型中，卷积神经网络能够更好的利用图像的结构信息。下面定义了一个较简单的卷积神经网络。显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后使用softmax分类作为输出层。

# ![%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE.png](attachment:%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE.png)

# 池化是非线性下采样的一种形式，主要作用是通过减少网络的参数来减小计算量，并且能够在一定程度上控制过拟合。通常在卷积层的后面会加上一个池化层。paddlepaddle池化默认为最大池化。是用不重叠的矩形框将输入层分成不同的区域，对于每个矩形框的数取最大值作为输出层
# ![%E6%B1%A0%E5%8C%96.png](attachment:%E6%B1%A0%E5%8C%96.png)

# In[6]:


def convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像
        filter_size=5,     # 滤波器的大小
        num_filters=20,    # filter 的数量。它与输出的通道相同
        pool_size=2,       # 池化层大小2*2
        pool_stride=2,     # 池化层步长
        act="relu")        # 激活类型
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


# In[7]:


#定义输入数据
data_shape = [3, 32, 32]
# 定义全局变量image和label
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 获取分类器，用cnn进行分类
predict =  convolutional_neural_network(images)


# In[8]:


# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label) # 交叉熵
avg_cost = fluid.layers.mean(cost)                            # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)       #使用输入和标签计算准确率


# 接着是定义优化方法，这次我们使用的是Adam优化方法，同时指定学习率为0.001。

# In[9]:


# 定义优化方法
optimizer =fluid.optimizer.Adam(learning_rate=0.001)#Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
optimizer.minimize(avg_cost)# 取最小的优化平均损失
print(type(acc))


# ![%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0.png](attachment:%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0.png)

# 训练分为三步：第一步配置好训练的环境，第二步用训练集进行训练，并用验证集对训练进行评估，不断优化，第三步保存好训练的模型

# In[10]:


# 使用CPU进行训练
place = fluid.CPUPlace()
# 创建一个executor
exe = fluid.Executor(place)
# 对program进行参数初始化1.网络模型2.损失函数3.优化函数
exe.run(fluid.default_startup_program())
# 定义输入数据的维度,DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
feeder = fluid.DataFeeder( feed_list=[images, label],place=place)


# 这次训练2个Pass。每一个Pass训练结束之后，再使用验证集进行验证，并求出相应的损失值Cost和准确率acc。

# In[11]:


# 训练的轮数
EPOCH_NUM = 10
# 开始训练
for pass_id in range(EPOCH_NUM):
    # 开始训练
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):              #遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(data),              #喂入一个batch的数据
                             fetch_list=[avg_cost, acc])          #fetch均方误差和准确率
        if batch_id % 100 == 0:                                   #每100次batch打印一次训练、进行一次测试
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 开始测试
    test_costs = []                                                        #测试的损失值
    test_accs = []                                                         #测试的准确率
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),#运行测试程序
                                      feed=feeder.feed(data),              #喂入一个batch的数据
                                      fetch_list=[avg_cost, acc])          #fetch均方误差、准确率
        test_costs.append(test_cost[0])                                    #记录每个batch的误差
        test_accs.append(test_acc[0])                                      #记录每个batch的准确率
    test_cost = (sum(test_costs) / len(test_costs))                        #计算误差平均值（误差和/误差的个数）
    test_acc = (sum(test_accs) / len(test_accs))                           #计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
    
    #保存模型
    model_save_dir = "/home/aistudio/data/catdog.inference.model"
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    fluid.io.save_inference_model(model_save_dir, ['images'], [predict], exe)
print('训练模型保存完成！')


# ![%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B.png](attachment:%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B.png)

# 下面是预测程序，直接单独运行In[*]就可以。预测主要有四步：第一步配置好预测的环境，第二步准备好要预测的图片，第三步加载预测的模型，把要预测的图片放到模型里进行预测，第四步输出预测的结果

# In[12]:


# 导入需要的包 nump: python第三方库，用于进行科学计算 PIL : Python Image Library,python,python第三方图像处理库
# matplotlib:python的绘图库 
# pyplot:matplotlib的绘图框架 

import numpy as np
from PIL import Image
import paddle.fluid as fluid
import matplotlib.pyplot as plt

# 使用CPU进行训练
place = fluid.CPUPlace()
# 定义一个executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()#要想运行一个网络，需要指明它运行所在的域，确切的说： exe.Run(&scope) 。
model_save_dir = "/home/aistudio/data/catdog.inference.model"

def load_image(file):
        #打开图片
        im = Image.open(file)
        #将图片调整为跟训练数据一样的大小  32*32，设定ANTIALIAS，即抗锯齿.resize是缩放
        im = im.resize((32, 32), Image.ANTIALIAS)
        #建立图片矩阵 类型为float32
        im = np.array(im).astype(np.float32)
        #矩阵转置 
        im = im.transpose((2, 0, 1))  # CHW，因为输入图像格式为[N，C，H，W]
        #将像素值从【0-255】转换为【0-1】
        im = im / 255.0

#         print(im)        # 保持和之前输入image维度一致
        im = np.expand_dims(im, axis=0)
        print('im_shape的维度：',im.shape)
        return im
#fluid.scope_guard修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope
with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载 推理model(inference model)
    [inference_program, # 预测用的program
     feed_target_names, # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
   
    img = Image.open('/home/aistudio/data/data2587/dog.png')
    plt.imshow(img)   #根据数组绘制图像
    plt.show()        #显示图像
    #获取推测数据，可以分别用猫和狗的图片进行预测
#     img = load_image( '/home/aistudio/data/data2587/cat .jpg')
    img = load_image( '/home/aistudio/data/data2587/dog.png')

    results = infer_exe.run(inference_program,                 #运行预测程序
                            feed={feed_target_names[0]: img},  #喂入要预测的img
                            fetch_list=fetch_targets)          #得到推测结果
    print('results',results)
    label_list = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
        ]
    print("infer results: %s" % label_list[np.argmax(results[0])])

