import os
import pandas
import shutil

#sort images to subfolders

data_dir = os.getcwd() + "/dataset/images/"
dest_dir = os.getcwd() + "/dataset/reorganized/"

#take images from "images" folder, put them into destination directiory into respective subfolders based on classes


#read the csv file
skin_df = pandas.read_csv('dataset/HAM10000_metadata.csv')
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