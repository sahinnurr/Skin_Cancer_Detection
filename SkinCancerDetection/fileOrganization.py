import os
import pandas
import shutil
import matplotlib.pyplot as plt

#sort images to subfolders

data_dir = os.getcwd() + "/dataset/images/"
dest_dir = os.getcwd() + "/dataset/reorganized/"

#take images from "images" folder, put them into destination directiory into respective subfolders based on classes


#read the csv file
skin_df = pandas.read_csv('dataset/HAM10000_metadata.csv')

print(skin_df.head()) #examine the beginning of the dataset
print(skin_df.info())  #Getting general information of the dataset
print(skin_df['dx'].value_counts())

label=skin_df['dx'].unique().tolist()  #Extract labels into a list
label_images = []

# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/") #creating a directory for each label
    sample = skin_df[skin_df['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg")) #copying images into corresponding subfolders
    label_images=[]


#showing sample images
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
        img_id = skin_df.iloc[i]['image_id']
        img_path = os.path.join(data_dir, img_id + ".jpg")
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

plt.show()