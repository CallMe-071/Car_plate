import cv2
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def detect(image):
    # 定义分类器
    cascade_path = 'cascade.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    # 修改图片大小
    resize_h = 400
    height = image.shape[0]
    scale = image.shape[1] / float(height)
    image_1 = cv2.resize(image, (int(scale * resize_h), resize_h))
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    # 转为灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 车牌定位
    car_plates = cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    print("检测到车牌数", len(car_plates))
    if len(car_plates) > 0:
        for car_plate in car_plates:
            x, y, w, h = car_plate
            # cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
            cv2.rectangle(image, (x , y ), (x + w , y + h ), (255, 0, 0), 1)
            test_img = image_1[y:(y + h), x:(x + w)]#截取图片要先选择高度再选择宽度
            test_img = cv2.resize(test_img,(400,140))
            cv2.imwrite("card_img_0.jpg", test_img)
    cv2.imshow("image", image)
    cv2.imshow("test_img", test_img)
#




# 分割图像
def find_end(start_):
  end_ = start_+1
  for m in range(start_+1, width-1):
    if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max): # 0.95这个参数请多调整，对应下面的0.05
      end_ = m
      break
  return end_




if __name__ == '__main__':
    car = cv2.imread('img/car_0.jpg')
    detect(car)
    image = cv2.imread('card_img_0.jpg')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    cv2.imshow('gray', img_gray)  # 显示图片
    cv2.waitKey(0)

    # 2、将灰度图像二值化，设定阈值是100
    img_thre = img_gray
    # cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_thre)
    cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY, img_thre)
    cv2.imshow('threshold', img_thre)
    cv2.waitKey(0)

    # 3、保存黑白图片
    # cv2.imwrite('thre_res.bmp', img_thre)

    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        # print(s)
        # print(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    n = 1
    start = 1
    end = 2
    index = 0
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = img_thre[1:height, start:end]
                single_img = cv2.resize(cj,(20,20))
                # single_img = cv2.cvtColor(single_img,cv2.COLOR_BGR2GRAY)
                cv2.imshow('single_img', single_img)
                cv2.imwrite('single_img_' + str(index) + '.bmp',single_img)
                index += 1
                cv2.waitKey(0)



