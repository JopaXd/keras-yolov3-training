from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import numpy as np
import urllib3
import shutil
import os
import shlex
import csv
import cv2

TRAIN_ANNOTATIONS = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
# V5 because this file is most likely the same for V6.
CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

TRAIN_ANNOTATIONS_FILE_NAME = TRAIN_ANNOTATIONS.split("/")[-1]
CLASS_DESCRIPTIONS_FILE_NAME = CLASS_DESCRIPTIONS.split("/")[-1]

DATA_PATH = "./data"
DATASET_PATH = "./data/dataset"

http = urllib3.PoolManager()

class ClassImage:

	def __init__(self, image_id=None, bounding_boxes=[], class_name=None, class_label=None, image_path=None):
		self.image_id = image_id
		self.bounding_boxes = []
		self.class_name = class_name
		self.class_label = class_label
		self.image_path = image_path

	def _convert_bounding_boxes(self):
		image = cv2.imread(self.image_path)
		for bbox in self.bounding_boxes:
			#XMin
			bbox[0] = str(int(float(bbox[0]) * int(image.shape[1])))
			#YMin
			bbox[1] = str(int(float(bbox[1]) * int(image.shape[0])))
			#XMax
			bbox[2] = str(int(float(bbox[2]) * int(image.shape[1])))
			#YMax
			bbox[3] = str(int(float(bbox[3]) * int(image.shape[0])))

	def _get_bounding_boxes_and_label_str(self):
		bboxes = ""
		for bbox in self.bounding_boxes:
			bboxes = bboxes + ",".join(bbox) + f",{self.class_label}"
			bboxes = bboxes + " "
		#The last character is a leftover space from the for loop, we get rid of that with [:-1].
		return bboxes[:-1]


	def __str__(self):
		return f"{self.image_path} {self._get_bounding_boxes_and_label_str()}"

def get_train_annotations():
	#Downloads the train annotations.
	with http.request('GET', TRAIN_ANNOTATIONS, preload_content=False) as r, open(f"{DATA_PATH}/{TRAIN_ANNOTATIONS_FILE_NAME}", 'wb') as out_file:
		shutil.copyfileobj(r, out_file)
	out_file.close()

def get_class_descriptions():
	#Downloads the class descriptions.
	with http.request('GET', CLASS_DESCRIPTIONS, preload_content=False) as r, open(f"{DATA_PATH}/{CLASS_DESCRIPTIONS_FILE_NAME}", 'wb') as out_file:
		shutil.copyfileobj(r, out_file)
	out_file.close()

def get_ids_from_class_names(classes):
	#Gets the ids that belong to the classes.
	ids = {}
	with open(f"{DATA_PATH}/{CLASS_DESCRIPTIONS_FILE_NAME}") as class_desc:
		csv_reader = csv.reader(class_desc, delimiter=",")
		for cclass in classes:
			for row in csv_reader:
				if row[1] == cclass:
					ids[row[1]] = row[0]
					break
	class_desc.close()
	return ids


def read_in_chunks(reader, chunksize=1024):
    chunk = []
    for i, line in enumerate(reader):
        if (i % chunksize == 0 and i > 0):
            yield chunk
            del chunk[:]
        chunk.append(line)
    yield chunk


def label_items(arr):
    vals,labels = np.unique(arr, return_inverse=True)
    return list(labels)

#Creates the text files necessary for training the model.
def create_train_files(img_list, class_list):
		with open("class_list.txt", "w") as classes_file:
			for c in class_list:
				classes_file.write(f"{c}\n")
		classes_file.close()
		with open("data.txt", "w") as data:
			for img in img_list:
				data.write(f"{str(img)}\n")
		data.close()

#Need to optimize this.
def download_images(ids, images_per_class, class_list):
	#Downloads the images for the dataset.
	img_list = []
	class_labels = label_items(class_list) 
	with open(f"{DATA_PATH}/{TRAIN_ANNOTATIONS_FILE_NAME}") as train_annotations:
		for img_class, class_id in ids.items():
			pool = ThreadPool(20)
			command_list = []
			img_count = 0
			csv_reader = csv.reader(train_annotations, delimiter=",")
			for chunk in read_in_chunks(csv_reader):
				if img_count == images_per_class:
					break
				for row in chunk:
					if img_count == images_per_class:
						break
					if row[2] == class_id:
						same_image = False
						#Processing has to be done here.
						XMin = row[4]
						XMax = row[5]
						YMin = row[6]
						YMax = row[7]
						image_id = row[0]
						for img in img_list:
							if img.image_id == image_id:
								img.bounding_boxes.append([XMin, YMin, XMax, YMax])
								same_image = True
								break
						if not same_image:
							new_img = ClassImage()
							new_img.image_id = image_id
							new_img.class_name = img_class
							new_img.class_label = class_labels[class_list.index(img_class)]
							new_img.image_path = os.path.abspath(f"{DATASET_PATH}/train/{img_class}/{image_id}.jpg")
							new_img.bounding_boxes.append([XMin, YMin, XMax, YMax])
							img_list.append(new_img)
							img_count+=1
							command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/train/{image_id}.jpg {DATASET_PATH}/train/{img_class}"
							command_list.append(command)
			list(tqdm(pool.imap(os.system, command_list), total = len(command_list) ))
			pool.close()
			pool.join()
	for img in img_list:
		#Make sure the coords for the bounding boxes are in correct format at the end.
		#For this to work, the images must be downloaded, that's why we're doing it here.
		img._convert_bounding_boxes()
	return img_list

def main():
	#Just make sure the data path exists.
	if not os.path.exists(DATA_PATH):
		os.mkdir(DATA_PATH)
	#Make sure the necessary files are downloaded.
	if not os.path.exists(f"{DATA_PATH}/{TRAIN_ANNOTATIONS_FILE_NAME}"):
		download_train_annotations = input(f"{TRAIN_ANNOTATIONS_FILE_NAME} is missing, would you like to download it? Y/N: ")
		if download_train_annotations.lower() == "y":
			print("Downloading...")
			get_train_annotations()
		else:
			print("File not downloaded! Cannot proceed! Exiting...")
			return
	if not os.path.exists(f"{DATA_PATH}/{CLASS_DESCRIPTIONS_FILE_NAME}"):
		download_class_annotations = input(f"{CLASS_DESCRIPTIONS_FILE_NAME} is missing, would you like to download it? Y/N: ")
		if download_train_annotations.lower() == "y":
			print("Downloading...")
			get_class_descriptions()
		else:
			print("File not downloaded! Cannot proceed! Exiting...")
			return
	#Creating the dataset.
	classes = input('List the classes you want to train (if a class contains multiple words in the name, use quotes, "Pencil sharpener"): ')
	if classes != "":
		class_list = shlex.split(classes)
		if not os.path.exists(f"{DATA_PATH}/dataset"):
			os.mkdir(f"{DATA_PATH}/dataset")
			os.mkdir(f"{DATA_PATH}/dataset/train")
		for cl in class_list:
			if not os.path.exists(f"{DATA_PATH}/dataset/train/{cl}"):
				os.mkdir(f"{DATA_PATH}/dataset/train/{cl}")
		ids = get_ids_from_class_names(class_list)
		num_of_images = int(input("How many images do you want to download per class?: "))
		print("Downloading images...")
		downloaded_images = download_images(ids, num_of_images, class_list)
		print("Creating train files...")
		create_train_files(downloaded_images, class_list)
		print("Done!")

if __name__ == "__main__":
	main()