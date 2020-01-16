import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


Pic_size = 400
Pic_width = 20
Pic_height = 20
Num = 6
input_learning_rate = 1e-4

Saver_dir = "train_saver_province/"

Provinces = ("京", "闽", "粤", "苏", "沪", "浙")


# 定义图像像素矩阵集合和对应的标签
x = tf.placeholder(tf.float32, shape=[None, Pic_size])
y_ = tf.placeholder(tf.float32, shape=[None, Num])


# 未知个width*height规格的图片数据对应的数组  方便求数组之间的乘积
x_image = tf.reshape(x, [-1, Pic_width, Pic_height, 1])


# 卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

if __name__ == '__main__':
    # 获取图片总数
    Pic_count = 0
    for i in range(0, Num):
        dir = 'train_images/training-set/chinese-characters/%s/' % i  # 将同一类型的图片存储在一个文件夹中进行遍历
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                Pic_count += 1

    # 定义对应维数和各维长度的数组
    input_images = np.array([[0] * Pic_size for i in range(Pic_count)])
    input_labels = np.array([[0] * Num for i in range(Pic_count)])

    # 生成训练集的图片数据和和标签
    index = 0
    for i in range(0, Num):
        dir = 'train_images/training-set/chinese-characters/%s/' % i  # 同上
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 使数字的线条变细，提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                input_labels[index][i] = 1
                index += 1

    # 获取验证集的图片数量
    Val_count = 0
    for i in range(0, Num):
        dir = 'train_images/validation-set/chinese-characters/%s/' % i  # 同上
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                Val_count += 1

    # 定义对应维数和各维长度的数组
    val_images = np.array([[0] * Pic_size for i in range(Val_count)])
    val_labels = np.array([[0] * Num for i in range(Val_count)])

    # 生成测试集的图片数据和标签
    index = 0
    for i in range(0, Num):
        dir = 'train_images/validation-set/chinese-characters/%s/' % i  # 同上
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 使数字的线条变细，提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w + h * width] = 0
                        else:
                            val_images[index][w + h * width] = 1
                val_labels[index][i] = 1
                index += 1

    with tf.Session() as sess:
        # 第一个卷积层
        Con1_W = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="Con1_W")
        Con1_b = tf.Variable(tf.constant(0.1, shape=[16]), name="Con1_b")
        # 卷积层的步幅长度
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, Con1_W, Con1_b, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        Con2_W = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="Con2_W")
        Con2_b = tf.Variable(tf.constant(0.1, shape=[32]), name="Con2_b")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, Con2_W, Con2_b, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        Fc1_W = tf.Variable(tf.truncated_normal([10 * 10 * 32, 512], stddev=0.1), name="Fc1_W")
        Fc1_b = tf.Variable(tf.constant(0.1, shape=[512]), name="Fc1_b")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 10 * 10 * 32])
        h_fc1 = full_connect(h_pool2_flat, Fc1_W, Fc1_b)

        # dropout正则化
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 输出层
        Fc2_W = tf.Variable(tf.truncated_normal([512, Num], stddev=0.1), name="Fc2_W")
        Fc2_b = tf.Variable(tf.constant(0.1, shape=[Num]), name="Fc2_b")

        # 定义优化器和训练op

        # 输出层函数的定义
        y_conv = tf.matmul(h_fc1_drop, Fc2_W) + Fc2_b
        # 求出对应省份标签所对应的函数值->平均值
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

        # 优化算法 反向传播的优化算法
        train_step = tf.train.AdamOptimizer((input_learning_rate)).minimize(cross_entropy)

        # 比较训练值和实际对应标签值对应的是否相等 相等为True
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # 求出True所占的比例及准确度
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化saver
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())


        print("总共有 %s 个训练图像， %s 个标签" % (Pic_count, Pic_count))

        # 规定一批的数据量
        Batch_size = 60
        # 规定迭代次数
        Iterations = 300
        # 获得数据量的总批次
        batches_count = int(Pic_count / Batch_size)
        # 若数据不为60的整数倍则 得出剩余数据数量
        remainder = Pic_count % Batch_size
        print("训练集总共分成 %s 批, 每批为 %s 个数据，其中最后一批为 %s 个数据" % (batches_count + 1, Batch_size, remainder))
        # 设置存储坐标的列表
        # 准确率横坐标
        picture_x = []
        # 准确率纵坐标
        picture_y = []
        # 损失函数横坐标
        pic_x = []
        # 损失函数纵坐标
        loss = []
        fig = plt.figure()
        # 横坐标的值
        index_x = 1
        # 进行迭代
        for it in range(Iterations):
            # 按批次的进行训练
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n * Batch_size:(n + 1) * Batch_size],
                                          y_: input_labels[n * Batch_size:(n + 1) * Batch_size], keep_prob: 0.5})
                loss.append(sess.run(cross_entropy,feed_dict={x: input_images[n * Batch_size:(n + 1) * Batch_size],
                                          y_: input_labels[n * Batch_size:(n + 1) * Batch_size], keep_prob: 0.5}))
                pic_x.append(index_x)
                index_x += 1


            if remainder > 0:
                start_index = batches_count * Batch_size;
                train_step.run(feed_dict={x: input_images[start_index:Pic_count - 1],
                                          y_: input_labels[start_index:Pic_count - 1], keep_prob: 0.5})

            # 完成5次迭代后判断准确率是否达到100%，是则退出迭代
            iterate_accuracy = 0
            if it % 5 == 0:
                picture_x.append(it)
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                picture_y.append(iterate_accuracy)

                print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it , iterate_accuracy * 100))
                if iterate_accuracy >= 0.9999 and it >= 150:
                    break;

        plt.subplot(2, 1, 1)
        plt.plot(picture_x, picture_y,color = 'red')
        plt.title('Province_Accuracy_Image')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(pic_x, loss,color = 'blue')
        plt.title('Province_Loss_Image')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        # 显示准确率图标
        plt.show()
        print('完成训练!')

        # 保存训练结果
        if not os.path.exists(Saver_dir):
            print('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(Saver_dir)
        saver_path = saver.save(sess, "%smodel.ckpt" % (Saver_dir))

