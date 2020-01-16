import cv2
import os
import numpy as np


# 根据图像的边界的像素值

for i in range(0,34):
    index = 100
    dir = 'train/%i/' %i # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = rt + filename
            img = cv2.imread(filename)
            img_detech = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
            img_1 = cv2.resize(img_detech,(20,20))
            img_out = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('train_other/%s/' %i + str(index)+'.jpg',img_out)
            index += 1


#
# # 常数填充

for i in range(0,34):
    index = 200
    dir = 'train/%s/' %i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = rt + filename
            img = cv2.imread(filename)
            img_detech = cv2.copyMakeBorder(img,1,1,1,1, cv2.BORDER_CONSTANT,value=[0,0,0])
            img_detech = cv2.resize(img_detech,(20,20))
            img_detech = cv2.cvtColor(img_detech,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('train_other/%s/' %i+str(index)+'.jpg',img_detech)
            index += 1




# 亮度和对比度调节

for m in range(0,34):
    index = 300
    dir = 'train/%s/' %m
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = rt + filename
            img = cv2.imread(filename)
            rows, cols, channels = img.shape
            dst = img.copy()
            a = 1.2
            b = 100
            for i in range(rows):
                for j in range(cols):
                    for c in range(3):
                        color = img[i, j][c] * a + b
                        if color > 255:
                            dst[i, j][c] = 255
                        elif color < 0:
                            dst[i, j][c] = 0

            img_1 = cv2.resize(dst,(20,20))
            img_out = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('train_other/%s/' %m +str(index)+'.jpg',img_out)
            index += 1


# 锐化


for i in range(0, 34):
    index = 400
    dir = 'train/%s/' % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = rt + filename
            img = cv2.imread(filename)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            dst = cv2.filter2D(img, -1, kernel=kernel)
            dst = cv2.resize(dst,(20,20))
            dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('train_other/%s/' %i + str(index)+'.jpg',dst)
            index += 1


# 模糊


for i in range(0, 34):
    index = 500
    dir = 'train/%s/' % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = rt + filename
            img = cv2.imread(filename)
            dst = cv2.blur(img, (15, 1))
            dst = cv2.resize(dst,(20,20))
            dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('train_other/%s/' %i +str(index)+'.jpg',dst)
            index += 1