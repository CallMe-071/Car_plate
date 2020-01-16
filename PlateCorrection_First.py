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
            # cv2.imwrite("plate_img/card_img.jpg", test_img)
    cv2.imshow("image", image)
    cv2.imshow("test_img", test_img)
    car_plate = test_img
    return car_plate



if __name__ == '__main__':
    image = cv2.imread('img/car_3.jpg')
    mid_img  = detect(image)
    cv2.imshow('mid_img',mid_img)
    cv2.imwrite('plate_img/carplate_3.jpg',cv2.resize(mid_img,(400,140)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()