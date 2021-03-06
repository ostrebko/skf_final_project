import numpy as np 
import os 
import glob 
import time
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from PIL import Image

root_folder = 'part_3_service'
path_to_images = os.path.join(root_folder, 'image_folder')
path_to_predicted_images = os.path.join(root_folder, 'predicted_images')
path_to_model_dir = os.path.join(root_folder, 'model')
path_to_yolo5 = os.path.join(root_folder, 'yolov5')

model_name = os.listdir(path_to_model_dir)[0]
path_to_model = os.path.join(path_to_model_dir, model_name)

if os.path.isdir(path_to_predicted_images):
    shutil.rmtree(path_to_predicted_images)
os.mkdir(path_to_predicted_images)


def create_model(path_to_model):
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path_to_model, device='cpu')
    model = torch.hub.load(path_to_yolo5, 'custom', path=path_to_model, source='local', 
                           device='cpu', force_reload=True)
    return model


model = create_model(path_to_model=path_to_model)


def calc_boxes(img_path, width=1280, height=1280, is_draw=False):
    
    """Для каждой фото рассчитывается количество бластоспор 
    (предсказание запускается на каждое фото). 
    Далее предсказанное количество бластоспор собраются в лист 
    и их число усредняется."""
    
    
    img_name = os.path.basename(img_path)


    original_image = Image.open(img_path)
    w, h = original_image.size

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    resized_image = original_image.resize(max_size, Image.ANTIALIAS)
    predict_boxes = model(resized_image, size=1280)    
    
    #image_numpy = cv2.imread(img_path)
    
    boxes = predict_boxes.xyxy[0][:, :4]
    numb_blasto = len(boxes)
    print(f'название: {img_name} число бластоспор: {numb_blasto}')
    
    if is_draw:
        plt.rcParams["figure.figsize"] = (12,8)
        fig, ax = plt.subplots()
        
        plt.imshow(resized_image)
        
        for item in boxes:
            width_img = int(item[2] - item[0])
            height_img = int(item[3] - item[1])
            ax = plt.gca()
            rect = patches.Rectangle((item[0],
                                      item[1]), 
                                     width_img, 
                                     height_img, 
                                     linewidth=1, 
                                     edgecolor='red', 
                                     fill = False)
            ax.add_patch(rect)
        
        save_file = os.path.join(path_to_predicted_images, img_name)
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close(fig) 
        
    return numb_blasto


def calc_avg_num_blastos(path_to_imgs, is_draw=True):
    start_time = time.time()
    imgs_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.jpg')), key=str)

    numb_images = len(imgs_paths)
    list_to_calc_avg = []
    #is_draw=True

    for num, imge in enumerate(imgs_paths, start=1):
        print(num, imge)
        print(f'Фото №{num}/{numb_images},', end=' ') 
        list_to_calc_avg.append(calc_boxes(imge, is_draw=is_draw))

    print()
    print(f'Среднее число бластоспор: {int(np.mean(list_to_calc_avg))}')

    print('Время расчета: %s seconds' % (time.time() - start_time))


calc_avg_num_blastos(path_to_images, is_draw=True)