from City_mode_train import *


if __name__ == '__main__':
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (Saver_dir))
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(Saver_dir)
        saver.restore(sess, model_file)

        # 第一个卷积层
        Con1_W = sess.graph.get_tensor_by_name("Con1_W:0")
        Con1_b = sess.graph.get_tensor_by_name("Con1_b:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, Con1_W, Con1_b, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        Con2_W = sess.graph.get_tensor_by_name("Con2_W:0")
        Con2_b = sess.graph.get_tensor_by_name("Con2_b:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, Con2_W, Con2_b, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        Fc1_W = sess.graph.get_tensor_by_name("Fc1_W:0")
        Fc1_b = sess.graph.get_tensor_by_name("Fc1_b:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 10 * 10 * 32])
        h_fc1 = full_connect(h_pool2_flat, Fc1_W, Fc1_b)

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout层
        Fc2_W = sess.graph.get_tensor_by_name("Fc2_W:0")
        Fc2_b = sess.graph.get_tensor_by_name("Fc2_b:0")

        # 定义优化器和训练op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, Fc2_W) + Fc2_b)


        path = "singleplate/7/1.jpg"
        img = Image.open(path)
        width = img.size[0]
        height = img.size[1]

        img_data = [[0] * Pic_size for i in range(1)]
        for h in range(0, height):
            for w in range(0, width):
                if img.getpixel((w, h)) < 190:
                    img_data[0][w + h * width] = 1
                else:
                    img_data[0][w + h * width] = 0

        result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

        max1 = 0
        max2 = 0
        max3 = 0
        max1_index = 0
        max2_index = 0
        max3_index = 0
        for j in range(Num):
            if result[0][j] > max1:
                max1 = result[0][j]
                max1_index = j
                continue
            if (result[0][j] > max2) and (result[0][j] <= max1):
                max2 = result[0][j]
                max2_index = j
                continue
            if (result[0][j] > max3) and (result[0][j] <= max2):
                max3 = result[0][j]
                max3_index = j
                continue

        license_num = license_num + LETTERS_DIGITS[max1_index]
        print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
        LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100, LETTERS_DIGITS[max3_index],
        max3 * 100))

    print("城市代号是: 【%s】" % license_num)
