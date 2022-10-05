from multiprocessing.dummy import Pool as ThreadPool
from itertools import islice
from tqdm import tqdm
import urllib3, shutil, os, shlex, cv2
import pandas as pd

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
	class_descriptions = pd.read_csv(f"{DATA_PATH}/{CLASS_DESCRIPTIONS_FILE_NAME}", header=None)
	rows = [list(x) for i, x in class_descriptions.iterrows()]
	for cclass in classes:
		for row in rows:
			if row[1] == cclass:
				ids[row[1]] = row[0]
				break
	return ids

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

def get_oidv_classes():
	class_descriptions = pd.read_csv(f"{DATA_PATH}/{CLASS_DESCRIPTIONS_FILE_NAME}", header=None)
	return class_descriptions[1].tolist()

#Downloads the images for the dataset.
def download_images(ids, images_per_class, class_list):
	img_list = []
	class_labels = list(range(0, len(class_list)))
	img_data = pd.read_csv(f"{DATA_PATH}/{TRAIN_ANNOTATIONS_FILE_NAME}", chunksize=1000000, header=None)
	already_downloaded_images = []
	num_of_images_to_download_for_class = {}
	for cl in class_list:
		num_of_images_to_download_for_class[cl] = images_per_class
		for image in os.listdir(f"{DATA_PATH}/dataset/train/{cl}"):
			already_downloaded_images.append(image.split(".")[0])
	command_list = []
	pool = ThreadPool(20)
	for chunk in img_data:
		for i, df_row in islice(chunk.iterrows(), 1, None):
			row = list(df_row)
			if row[2] not in list(ids.values()):
				continue
			for img_class, class_id in ids.items():
				if num_of_images_to_download_for_class[img_class] == 0:
					continue
				if row[2] == class_id:
					same_image = False
					XMin = row[4]
					XMax = row[5]
					YMin = row[6]
					YMax = row[7]
					image_id = row[0]
					for img in img_list:
						#Same image, just bounding box for another item in the image.
						if img.image_id == image_id:
							img.bounding_boxes.append([XMin, YMin, XMax, YMax])
							same_image = True
							break
					if not same_image:
						#New image, add it to the list, and check if its already downloaded.
						new_img = ClassImage()
						new_img.image_id = image_id
						new_img.class_name = img_class
						new_img.class_label = class_labels[class_list.index(img_class)]
						new_img.image_path = os.path.abspath(f"{DATASET_PATH}/train/{img_class}/{image_id}.jpg")
						new_img.bounding_boxes.append([XMin, YMin, XMax, YMax])
						img_list.append(new_img)
						num_of_images_to_download_for_class[img_class]-=1
						print(num_of_images_to_download_for_class[img_class])
						if image_id not in already_downloaded_images:
							#Image has not been downloaded already.
							command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/train/{image_id}.jpg {DATASET_PATH}/train/{img_class}"
							command_list.append(command)
		print("a")
		if sum(list(num_of_images_to_download_for_class.values())) == 0:
			#All images have been added.
			break
	if len(command_list) != 0:
		list(tqdm(pool.imap(os.system, command_list), total = len(command_list) ))
	pool.close()
	pool.join()
	for img in img_list:
		#Make sure the coords for the bounding boxes are in correct format at the end.
		#For this to work, the images must be downloaded, that's why we're doing it here.
		img._convert_bounding_boxes()
	return img_list

def main():
	#Make sure AWS CLI is installed.
	if not shutil.which("aws"):
		print("AWS CLI is not installed or not in PATH! Please install AWS CLI and add it to PATH to proceed.")
		return
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
		#Probably an overthought way of making the first character upper-case for every class.
		class_list = [x[0].upper() + x[1:] for x in class_list]
		oidv_classes = get_oidv_classes()
		#Check for invalid classes.
		invalid_classes = []
		for cl in class_list:
			if cl not in oidv_classes:
				invalid_classes.append(cl)
		if len(invalid_classes) > 0:
			print(f"The following classes are invalid: {' '.join(invalid_classes)}")
			for inv_cl in invalid_classes:
				if inv_cl in class_list:
					class_list.remove(inv_cl)
		if len(class_list) == 0:
			print("No valid classes were listed! Exiting...")
			return
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