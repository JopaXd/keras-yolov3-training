from itertools import dropwhile, takewhile
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import urllib3
import shutil
import os
import shlex
import csv

TRAIN_ANNOTATIONS = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
# V5 because this file is most likely the same for V6.
CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

TRAIN_ANNOTATIONS_FILE_NAME = TRAIN_ANNOTATIONS.split("/")[-1]
CLASS_DESCRIPTIONS_FILE_NAME = CLASS_DESCRIPTIONS.split("/")[-1]

DATA_PATH = "./data"
DATASET_PATH = "./data/dataset"

http = urllib3.PoolManager()

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
		for row in csv_reader:
			if row[1] in classes:
				ids[row[1]] = row[0]
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


#Need to optimize this.
def download_images(ids, images_per_class):
	#Downloads the images for the dataset.
	with open(f"{DATA_PATH}/{TRAIN_ANNOTATIONS_FILE_NAME}") as train_annotations:
		for img_class, class_id in ids.items():
			pool = ThreadPool(5)
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
						XMin = row[4]
						XMax = row[5]
						YMin = row[6]
						YMax = row[7]
						image_id = row[0]
						img_count+=1
						command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/train/{image_id}.jpg {DATASET_PATH}/train/{img_class}"
						command_list.append(command)
			list(tqdm(pool.imap(os.system, command_list), total = len(command_list) ))
			pool.close()
			pool.join()

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
		download_images(ids, num_of_images)

if __name__ == "__main__":
	main()