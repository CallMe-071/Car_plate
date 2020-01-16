import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



Pic_size = 400
Pic_width = 20
Pic_height = 20
Num = 34


Saver_dir = "train_saver_license/"

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
license_num = ""


# 定义图像像素矩阵集合和对应的标签
x = tf.placeholder(tf.float32, shape=[None, Pic_size])
y_ = tf.placeholder(tf.float32, shape=[None, Num])

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
    # 第一次遍历图片目录是为了获取图片总数
    input_count = 0
    for i in range(0, Num):
        dir = 'train_images/training-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1

    # 定义对应维数和各维长度的数组
    input_images = np.array([[0] * Pic_size for i in range(input_count)])
    input_labels = np.array([[0] * Num for i in range(input_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0, Num):
        dir = 'train_images/training-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                input_labels[index][i] = 1
                index += 1

    # 第一次遍历图片目录是为了获取图片总数
    val_count = 0
    for i in range(0, Num):
        dir = 'train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1

    # 定义对应维数和各维长度的数组
    val_images = np.array([[0] * Pic_size for i in range(val_count)])
    val_labels = np.array([[0] * Num for i in range(val_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0, Num):
        dir = 'train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
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

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout层
        Fc2_W = tf.Variable(tf.truncated_normal([512, Num], stddev=0.1), name="Fc2_W")
        Fc2_b = tf.Variable(tf.constant(0.1, shape=[Num]), name="Fc2_b")

        # 定义优化器和训练op
        y_conv = tf.matmul(h_fc1_drop, Fc2_W) + Fc2_b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())


        print("一共读取了 %s 个训练图像， %s 个标签" % (input_count, input_count))

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        Iterations = 1000
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print("训练数据集分成 %s 批, 每批 %s 个数据，其中最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

        picture_x = []
        picture_y = []
        loss = []
        pic_x = []
        fig = plt.figure()
        index_x = 1


        # 执行训练迭代
        for it in range(Iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                          y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
                loss.append(sess.run(cross_entropy, feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                                               y_: input_labels[n * batch_size:(n + 1) * batch_size],
                                                               keep_prob: 0.5}))
                pic_x.append(index_x)
                index_x += 1
            if remainder > 0:
                start_index = batches_count * batch_size;
                train_step.run(feed_dict={x: input_images[start_index:input_count - 1],
                                          y_: input_labels[start_index:input_count - 1], keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it % 5 == 0:
                picture_x.append(it)
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})

                picture_y.append(iterate_accuracy)

                print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
                if iterate_accuracy >= 0.9999 and it >= 250:
                    break;

        plt.subplot(2, 1, 1)
        plt.plot(picture_x, picture_y, color='red')
        plt.title('License_Accuracy_Image')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(pic_x, loss, color='blue')
        plt.title('License_Loss_Image')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        # 显示准确率图标
        plt.show()

        print('完成训练!')


        # 保存训练结果
        if not os.path.exists(Saver_dir):
            print('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(Saver_dir)
        # 初始化saver
        saver = tf.train.Saver()
        saver_path = saver.save(sess, "%smodel.ckpt" % (Saver_dir))