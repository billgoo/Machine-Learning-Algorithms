
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录
get_ipython().system('ls /home/aistudio/data/')


# In[25]:


# 查看个人持久化工作区文件
get_ipython().system('ls /home/aistudio/work/')
import os
from PIL import Image # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid
from __future__ import print_function # 将python3中的print特性导入当前版本


# In[26]:


def softmax_regression():
    """
    定义softmax分类器：
        一个以softmax为激活函数的全连接层
    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 以softmax为激活函数的全连接层，输出层的大小必须为数字的个数10
    predict = fluid.layers.fc(
        input=img, size=10, act='softmax')
    return predict


# In[27]:


def multilayer_perceptron():
    """
    定义多层感知机分类器：
        含有两个隐藏层（全连接层）的多层感知器
        其中前两个隐藏层的激活函数采用 ReLU，输出层的激活函数用 Softmax
    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


# In[28]:


def convolutional_neural_network():
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层
    Return:
        predict -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    # 使用50个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


# In[29]:


def train_program():
    """
    配置train_program
    Return:
        predict -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率
    """
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    
    # predict = softmax_regression() # 取消注释将使用 Softmax回归
    # predict = multilayer_perceptron() # 取消注释将使用 多层感知器
    predict = convolutional_neural_network() # 取消注释将使用 LeNet5卷积神经网络
    
    # 使用类交叉熵函数计算predict和label之间的损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 计算平均损失
    avg_cost = fluid.layers.mean(cost)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]


# In[30]:


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


# In[31]:


# 一个minibatch中有64个数据
BATCH_SIZE = 64
# 每次读取训练集中的500个数据并随机打乱，传入batched reader中，batched reader 每次 yield 64个数据
train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
# 读取测试集的数据，每次 yield 64个数据
test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)


# In[32]:


def event_handler(pass_id, batch_id, cost):
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass %d, Batch %d, Cost %f" % (pass_id,batch_id, cost))


# In[33]:


from paddle.utils.plot import Ploter
train_prompt = "Train cost"
test_prompt = "Test cost"
cost_ploter = Ploter(train_prompt, test_prompt)
# 将训练过程绘图表示
def event_handler_plot(ploter_title, step, cost):
    cost_ploter.append(ploter_title, step, cost)
    cost_ploter.plot()


# In[34]:


# 该模型运行在单个CPU上
use_cuda = False # 如想使用GPU，请设置为 True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 调用train_program 获取预测值，损失值，
prediction, [avg_loss, acc] = train_program()
# 输入的原始图像数据，大小为28*28*1
img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
# 标签层，名称为label,对应输入图片的类别标签
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
# 选择Adam优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_loss)


# In[35]:


PASS_NUM = 5 #训练5轮
epochs = [epoch_id for epoch_id in range(PASS_NUM)]
# 将模型参数存储在名为 save_dirname 的文件中
save_dirname = "recognize_digits.inference.model"


# In[36]:


def train_test(train_test_program,
                    train_test_feed, train_test_reader):
    # 将分类准确率存储在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=[acc, avg_loss])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean


# In[37]:


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# In[38]:


main_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)


# In[39]:


lists = []
step = 0
for epoch_id in epochs:
    #print(epoch_id)
    for step_id, data in enumerate(train_reader()):
        metrics = exe.run(main_program,
                        feed=feeder.feed(data),
                        fetch_list=[avg_loss, acc])
        if step % 100 == 0: #每训练100次 打印一次log
            print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))
            event_handler_plot(train_prompt, step, metrics[0])
        step += 1
        # 测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                            train_test_reader=test_reader,
                                            train_test_feed=feeder)
        print("Test with Epoch %d, avg_cost: %s, acc: %s" %(epoch_id, avg_loss_val, acc_val))
        event_handler_plot(test_prompt, step, metrics[0])
        lists.append((epoch_id, avg_loss_val, acc_val))
        # 保存训练好的模型参数用于预测
        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname,
                                            ["img"], [prediction], exe,
                                            model_filename=None,
                                            params_filename=None)
# 选择效果最好的pass
best = sorted(lists, key=lambda list: float(list[1]))[0]
print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

