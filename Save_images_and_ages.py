import os
import csv
import numpy as np
import cv2
import pydicom
import nibabel as nib
import re


def load_images_with_ids(image_folder):
    images = []
    image_ids = []
    
    for root, _, files in os.walk(image_folder):
        folder_id = os.path.basename(root)  
        for file in files:
            if file.endswith(('.dcm', '.mnc')):
                file_path = os.path.join(root, file)
                if re.search(r'.*dcm$',file_path)!=None:               
                    
                    # Load MNC image
                    img=pydicom.dcmread(file_path)
                    img=img.pixel_array
                else:
                    
                    # Load DCM image
                    img=nib.load(file_path)
                    img=img.get_fdata()
                    img=img[:,:,img.shape[2]//2]
                
                img=cv2.resize(img,(224,224))
                img=img.astype('uint8')               
                images.append(img)

                # Find image id in its name
                parts=root.split('\\') 
                image_id=parts[-4]
                image_ids.append(image_id)
    return np.array(images), image_ids


def load_labels_from_csv(csv_file):
    labels = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_id = row['Subject']  
            label = row['Age']  
            labels[image_id] = label
    return labels


def allocate_labels_to_images(images, image_ids, labels_dict):
    image_labels = []
    
    for image_id in image_ids:
        if image_id in labels_dict:
            image_labels.append(labels_dict[image_id])
        else:
            image_labels.append(None)  
    
    return np.array(image_labels)


def main(image_folder, csv_file):
    # Load images and their IDs
    images, image_ids = load_images_with_ids(image_folder)
    labels_dict = load_labels_from_csv(csv_file)
    image_labels = allocate_labels_to_images(images, image_ids, labels_dict)
    return images, image_labels



image_folder='./ICBM T1-Weighted/ICBM'
csv_file='./ICBM_T1_Labels.csv'
images, image_labels = main(image_folder, csv_file)


# Save images and labels
saving_path='./Converted Images/'
counter=1
for image,label in zip(images,image_labels):
    cv2.imwrite(saving_path+f'ImageAge{label}_id{counter}.png',image)
    counter+=1


print("Images array shape:", images.shape)
print("Labels array shape:", image_labels.shape)



