import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore.train.model import Model
from src.config import config_gpu
from src.mobilenetV3 import mobilenet_v3_large
from mindspore.dataset.vision import Inter
import cv2


#加载待预测图像的真实分类标签
def read_label(label_path):
    label_list = []
    with open(label_path) as file:
        reads = file.readlines()
        for i in range(0, len(reads)):
            read = reads[i].strip('\n')
            label_list.append(int(read[:]))

    return label_list



#读取单张图像并进行预处理,用于predict
def data_test_handle(image_path):

    img = cv2.imread(image_path)
    # 1、类型转换
    # type_cast_op = C2.TypeCast(mstype.int32)

    # 2、resize大小缩放
    # resize_op = C.Resize(size=scale_size, interpolation=interpolation)
    scale_size = 256
    img = cv2.resize(img, (scale_size, scale_size), interpolation=Inter.BICUBIC)
    # 3、中心剪切成224*224
    # center_crop = C.CenterCrop(size=img_size)
    img = img[16:240,16:240]
    # 4、图像取值规模缩放：/255至0-1
    # rescale_op = C.Rescale(rescale, shift)
    img = np.array(img, dtype='float32')
    img /= 255.
    # 5、归一化
    # normalize_op = C.Normalize(config.mean,config.std)
    # 6、通道顺序变换 changeswap_op = C.HWC2CHW()
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # 将图像转成向量
    img_tensor = Tensor(img, dtype=mindspore.float32)

    return img_tensor



scale_size = 256
path_root = r'F:\Algorithm_Project_Code\ShengSi_challenge\Data\caltech_256\caltech_for_user\train\40'
best_ckpt_path = "F:\Algorithm_Project_Code\ShengSi_challenge\mode_ckpt\mobilenetV3\mobilenetV3-160_135.ckpt"

#acc = 0.2525


if __name__ == '__main__':
    net = mobilenet_v3_large(num_classes=config_gpu.num_classes)

    # 加载模型参数
    param_dict = mindspore.load_checkpoint(best_ckpt_path)
    mindspore.load_param_into_net(net, param_dict)
    model = Model(net)

    label_path = './400_799.txt'
    label_list = read_label(label_path)
    print(label_list)

    a = 0
    count = 0
    data_sum = len(label_list)
    # 打开文件
    #Note = open('./result/predict_private_datare.txt', mode='w')


    '''
    for i in range(400, 800):
        image_name = '\\' + str(i) + '.jpg'
        image_path = path_root + image_name
        temp = data_test_handle(image_path)
        # 开始预测
        predictions = model.predict(temp).asnumpy()
        r, c = np.where(predictions == np.max(predictions))
        c = c[0] + 1
        if c == label_list[a]:
            print('True ',"第", i, "个图像预测正确，label值为：", c)
            Note.write('True'+ ' '+ str(label_list[a]) + ' ' + str(c) + '\n')  # \n 换行符
            count = count + 1
        else:
            print('False ',"第", i, "个图像预测错误，label值为：", label_list[a], "预测值为：", c)
            Note.write('False' + ' ' + str(label_list[a]) + ' ' + str(c) + '\n')

        a = a + 1
    acc = count/data_sum
    print('\n\n','模型正确率为：', acc)

    Note.close()
    '''
    for i in range(2858,2901):
        image_name = '\\' + str(i) + '.jpg'
        image_path = path_root + image_name
        temp = data_test_handle(image_path)
        # 开始预测
        predictions = model.predict(temp).asnumpy()
        r, c = np.where(predictions == np.max(predictions))
        c = c[0] + 1
        print(c)
