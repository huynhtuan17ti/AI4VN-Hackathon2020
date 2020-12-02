import cv2
import os
import pandas as pd

IMAGE_PATH = "/home/ryan/Machine_Learning/AI4VN-Final/test_set"
SAVE_PATH = "/home/ryan/Machine_Learning/AI4VN-Final/test"

def process():
    error_image = []
    image_id = []
    lst = os.listdir(IMAGE_PATH)
    for img in sorted(lst):
        name_img = img[:-4]
        path = os.path.join(IMAGE_PATH, img)
        try:
            type_img = path[-4:]
            if type_img == 'jpeg':
                type_img = '.jpeg'
            if type_img == '.bmp':
                print("Error: ", img)
                error_image.append(img)
                continue
            img_array = cv2.imread(path)
            h, w, _ = img_array.shape
            #cv2.imwrite(os.path.join(SAVE_PATH , img), img_array)
            image_id.append(img)
        except Exception as E:
            print(E)
            error_image.append(img)
            print('Error {}'.format(img))
            pass
    return image_id, error_image

def to_error_csv(error_image):
    print('Found {} error images'.format(len(error_image)))
    dict = {"error_image": error_image}
    df = pd.DataFrame(dict)
    df.to_csv("error.csv", index = False)
    print('Make error.csv file sucessfully !')

def to_test_csv(image_id):
    print('Found {} available images'.format(len(image_id)))
    dict = {"image_id": image_id}
    df = pd.DataFrame(dict)
    df.to_csv("test.csv", index = False)
    print('Make test.csv file sucessfully !')

def main():
    image_id, error_image = process()
    to_error_csv(error_image)
    to_test_csv(image_id)
    

if __name__ == '__main__':
    main()
