import cv2


# 求出每列的像素总和
def column_sum(img, col):
    height = img.shape[0]
    sum = 0
    for i in range(height):
        sum += img[i, col]

    return sum


def cut(img, aver_sum, left):
    # left = 0
    for i in range(left, img.shape[1]):
        # 求出当前列的像素总和
        colValue = column_sum(img, i)
        # 如果当前列的像素总和大于平均列像素则表示存在字体 标记起始点
        if (colValue > aver_sum):
            left = i
            break
    # 单列的平均宽度
    single_width = img.shape[1] / 10
    # 如果当前列的像素总额和小于平均列像素则表示不存在字体 标记终点
    for i in range(left, img.shape[1]):
        colValue = column_sum(img, i)
        if colValue < aver_sum:
            right = i
            # 若总长度小于平均长度则不符合要求
            if (right - left) < single_width:
                continue
            else:
                break
            #     single_width = right - left
            #     break
    return left, right


if __name__ == '__main__':
    # 读取车牌号图片
    imge = cv2.imread('plate_img/carplate_7.jpg')
    # 灰度化
    img_gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # 行（高度）
    row = img_gray.shape[0]
    # 列（宽度）
    column = img_gray.shape[1]

    total_sum = 0
    # 求出整个图片的像素总和
    for i in range(column):
        total_sum += column_sum(img_binary, i)

    # 表示每一列的平均像素
    aver_sum = 0.6 * (total_sum / column)
    # 左起点
    left = 0
    # 右终点
    right = 0
    left, right = cut(img_binary, aver_sum, left)
    cv2.imshow('img', img_binary[0:140, left:right])
    cv2.imwrite('singleplate/7/0.jpg',cv2.resize(img_binary[0:140,left:right],(20,20)))
    cv2.waitKey(0)
    index = 1
    while (column - right) > (img_binary.shape[1] / 10):
        try:

            left = right
            left, right = cut(img_binary, aver_sum, left)
            cv2.imshow('img', img_binary[0:140, left:right])
            cv2.imwrite('singleplate/7/%s.jpg' %index,cv2.resize(img_binary[0:140,left:right],(20,20)))
            index += 1
            cv2.waitKey(0)
        except:
            break
