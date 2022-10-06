from keras.layers import Input, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import urllib3, shutil, os, shlex, cv2, configparser, io
import pandas as pd
import numpy as np
import keras.backend as K
import configparser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_ANNOTATIONS = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
# V5 because this file is most likely the same for V6.
CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

YOLO_WEIGHTS = "https://pjreddie.com/media/files/yolov3.weights"

TRAIN_ANNOTATIONS_FILE_NAME = TRAIN_ANNOTATIONS.split("/")[-1]
CLASS_DESCRIPTIONS_FILE_NAME = CLASS_DESCRIPTIONS.split("/")[-1]
YOLO_WEIGHTS_FILE_NAME = YOLO_WEIGHTS.split("/")[-1]

DATA_PATH = "./data"
DATASET_PATH = "./data/dataset"
MODEL_DATA_PATH = "./model_data"

KERAS_MODEL_PATH = f"{MODEL_DATA_PATH}/yolo_weights.h5"

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

def get_classes(classes_path):
	'''loads the classes'''
	with open(classes_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names

def get_anchors(anchors_path):
	'''loads the anchors from a file'''
	with open(anchors_path) as f:
		anchors = f.readline()
	anchors = [float(x) for x in anchors.split(',')]
	return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
	weights_path='model_data/yolo_weights.h5'):
	'''create the training model'''
	K.clear_session() # get a new session
	image_input = Input(shape=(None, None, 3))
	h, w = input_shape
	num_anchors = len(anchors)

	y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
		num_anchors//3, num_classes+5)) for l in range(3)]

	model_body = yolo_body(image_input, num_anchors//3, num_classes)
	print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

	if load_pretrained:
		model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
		print('Load weights {}.'.format(weights_path))
		if freeze_body in [1, 2]:
			# Freeze darknet53 body or freeze all but 3 output layers.
			num = (185, len(model_body.layers)-3)[freeze_body-1]
			for i in range(num): model_body.layers[i].trainable = False
			print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

	model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
		arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
		[*model_body.output, *y_true])
	model = Model([model_body.input, *y_true], model_loss)

	return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
			weights_path='model_data/tiny_yolo_weights.h5'):
	'''create the training model, for Tiny YOLOv3'''
	K.clear_session() # get a new session
	image_input = Input(shape=(None, None, 3))
	h, w = input_shape
	num_anchors = len(anchors)

	y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
		num_anchors//2, num_classes+5)) for l in range(2)]

	model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
	print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

	if load_pretrained:
		model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
		print('Load weights {}.'.format(weights_path))
		if freeze_body in [1, 2]:
			# Freeze the darknet body or freeze all but 2 output layers.
			num = (20, len(model_body.layers)-2)[freeze_body-1]
			for i in range(num): model_body.layers[i].trainable = False
			print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

	model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
		arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
		[*model_body.output, *y_true])
	model = Model([model_body.input, *y_true], model_loss)

	return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
	'''data generator for fit_generator'''
	n = len(annotation_lines)
	i = 0
	while True:
		image_data = []
		box_data = []
		for b in range(batch_size):
			if i==0:
				np.random.shuffle(annotation_lines)
			image, box = get_random_data(annotation_lines[i], input_shape, random=True)
			image_data.append(image)
			box_data.append(box)
			i = (i+1) % n
		image_data = np.array(image_data)
		box_data = np.array(box_data)
		y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
		yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
	n = len(annotation_lines)
	if n==0 or batch_size<=0: return None
	return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

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

def get_yolo_weights():
	#Downloads the YOLO weights file.
	with http.request('GET', YOLO_WEIGHTS, preload_content=False) as r, open(f"{MODEL_DATA_PATH}/{YOLO_WEIGHTS_FILE_NAME}", 'wb') as out_file:
		shutil.copyfileobj(r, out_file)
	out_file.close()

def convert_to_keras_model():
	config_path = f"{MODEL_DATA_PATH}/yolov3.cfg"
	weights_path = f"{MODEL_DATA_PATH}/{YOLO_WEIGHTS_FILE_NAME}"

	# Load weights and config.
	print('Loading weights.')
	weights_file = open(weights_path, 'rb')
	major, minor, revision = np.ndarray(
		shape=(3, ), dtype='int32', buffer=weights_file.read(12))
	if (major*10+minor)>=2 and major<1000 and minor<1000:
		seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
	else:
		seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
	print('Weights Header: ', major, minor, revision, seen)

	print('Parsing Darknet config.')
	unique_config_file = unique_config_sections(config_path)
	cfg_parser = configparser.ConfigParser()
	cfg_parser.read_file(unique_config_file)

	print('Creating Keras model.')
	input_layer = Input(shape=(None, None, 3))
	prev_layer = input_layer
	all_layers = []

	weight_decay = float(cfg_parser['net_0']['decay']
						 ) if 'net_0' in cfg_parser.sections() else 5e-4
	count = 0
	out_index = []
	for section in cfg_parser.sections():
		print('Parsing section {}'.format(section))
		if section.startswith('convolutional'):
			filters = int(cfg_parser[section]['filters'])
			size = int(cfg_parser[section]['size'])
			stride = int(cfg_parser[section]['stride'])
			pad = int(cfg_parser[section]['pad'])
			activation = cfg_parser[section]['activation']
			batch_normalize = 'batch_normalize' in cfg_parser[section]

			padding = 'same' if pad == 1 and stride == 1 else 'valid'

			# Setting weights.
			# Darknet serializes convolutional weights as:
			# [bias/beta, [gamma, mean, variance], conv_weights]
			prev_layer_shape = K.int_shape(prev_layer)

			weights_shape = (size, size, prev_layer_shape[-1], filters)
			darknet_w_shape = (filters, weights_shape[2], size, size)
			weights_size = np.product(weights_shape)

			print('conv2d', 'bn'
				  if batch_normalize else '  ', activation, weights_shape)

			conv_bias = np.ndarray(
				shape=(filters, ),
				dtype='float32',
				buffer=weights_file.read(filters * 4))
			count += filters

			if batch_normalize:
				bn_weights = np.ndarray(
					shape=(3, filters),
					dtype='float32',
					buffer=weights_file.read(filters * 12))
				count += 3 * filters

				bn_weight_list = [
					bn_weights[0],  # scale gamma
					conv_bias,  # shift beta
					bn_weights[1],  # running mean
					bn_weights[2]  # running var
				]

			conv_weights = np.ndarray(
				shape=darknet_w_shape,
				dtype='float32',
				buffer=weights_file.read(weights_size * 4))
			count += weights_size

			# DarkNet conv_weights are serialized Caffe-style:
			# (out_dim, in_dim, height, width)
			# We would like to set these to Tensorflow order:
			# (height, width, in_dim, out_dim)
			conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
			conv_weights = [conv_weights] if batch_normalize else [
				conv_weights, conv_bias
			]

			# Handle activation.
			act_fn = None
			if activation == 'leaky':
				pass  # Add advanced activation later.
			elif activation != 'linear':
				raise ValueError(
					'Unknown activation function `{}` in section {}'.format(
						activation, section))

			# Create Conv2D layer
			if stride>1:
				# Darknet uses left and top padding instead of 'same' mode
				prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
			conv_layer = (Conv2D(
				filters, (size, size),
				strides=(stride, stride),
				kernel_regularizer=l2(weight_decay),
				use_bias=not batch_normalize,
				weights=conv_weights,
				activation=act_fn,
				padding=padding))(prev_layer)

			if batch_normalize:
				conv_layer = (BatchNormalization(
					weights=bn_weight_list))(conv_layer)
			prev_layer = conv_layer

			if activation == 'linear':
				all_layers.append(prev_layer)
			elif activation == 'leaky':
				act_layer = LeakyReLU(alpha=0.1)(prev_layer)
				prev_layer = act_layer
				all_layers.append(act_layer)

		elif section.startswith('route'):
			ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
			layers = [all_layers[i] for i in ids]
			if len(layers) > 1:
				print('Concatenating route layers:', layers)
				concatenate_layer = Concatenate()(layers)
				all_layers.append(concatenate_layer)
				prev_layer = concatenate_layer
			else:
				skip_layer = layers[0]  # only one layer to route
				all_layers.append(skip_layer)
				prev_layer = skip_layer

		elif section.startswith('maxpool'):
			size = int(cfg_parser[section]['size'])
			stride = int(cfg_parser[section]['stride'])
			all_layers.append(
				MaxPooling2D(
					pool_size=(size, size),
					strides=(stride, stride),
					padding='same')(prev_layer))
			prev_layer = all_layers[-1]

		elif section.startswith('shortcut'):
			index = int(cfg_parser[section]['from'])
			activation = cfg_parser[section]['activation']
			assert activation == 'linear', 'Only linear activation supported.'
			all_layers.append(Add()([all_layers[index], prev_layer]))
			prev_layer = all_layers[-1]

		elif section.startswith('upsample'):
			stride = int(cfg_parser[section]['stride'])
			assert stride == 2, 'Only stride=2 supported.'
			all_layers.append(UpSampling2D(stride)(prev_layer))
			prev_layer = all_layers[-1]

		elif section.startswith('yolo'):
			out_index.append(len(all_layers)-1)
			all_layers.append(None)
			prev_layer = all_layers[-1]

		elif section.startswith('net'):
			pass

		else:
			raise ValueError(
				'Unsupported section header type: {}'.format(section))

	# Create and save model.
	if len(out_index)==0: out_index.append(len(all_layers)-1)
	model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
	model.save('{}'.format(KERAS_MODEL_PATH))
	print('Saved Keras model to {}'.format(KERAS_MODEL_PATH))

	# Check to see if all weights have been read.
	remaining_weights = len(weights_file.read()) / 4
	weights_file.close()
	print('Read {} of {} from Darknet weights.'.format(count, count +
													   remaining_weights))
	if remaining_weights > 0:
		print('Warning: {} unused weights'.format(remaining_weights))

def unique_config_sections(config_file):
	"""Convert all config sections to have unique names.

	Adds unique suffixes to config sections for compability with configparser.
	"""
	section_counters = defaultdict(int)
	output_stream = io.StringIO()
	with open(config_file) as fin:
		for line in fin:
			if line.startswith('['):
				section = line.strip().strip('[]')
				_section = section + '_' + str(section_counters[section])
				section_counters[section] += 1
				line = line.replace(section, _section)
			output_stream.write(line)
	output_stream.seek(0)
	return output_stream

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
						if image_id not in already_downloaded_images:
							#Image has not been downloaded already.
							command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/train/{image_id}.jpg {DATASET_PATH}/train/{img_class}"
							command_list.append(command)
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

def train_model():
	annotation_path = 'data.txt'
	log_dir = 'logs/000/'
	classes_path = 'class_list.txt'
	anchors_path = 'model_data/yolo_anchors.txt'
	class_names = get_classes(classes_path)
	num_classes = len(class_names)
	anchors = get_anchors(anchors_path)

	input_shape = (416,416) # multiple of 32, hw

	is_tiny_version = len(anchors)==6 # default setting
	if is_tiny_version:
		model = create_tiny_model(input_shape, anchors, num_classes,
			freeze_body=2, weights_path='model_data/yolo_weights.h5')
	else:
		model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze


	logging = TensorBoard(log_dir=log_dir)
	checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
		monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

	val_split = 0.1
	with open(annotation_path) as f:
		lines = f.readlines()
	np.random.shuffle(lines)
	num_val = int(len(lines)*val_split)
	num_train = len(lines) - num_val

	print("Training with frozen layers...")

	model.compile(optimizer=Adam(lr=1e-3), loss={
		# use custom yolo_loss Lambda layer.
		'yolo_loss': lambda y_true, y_pred: y_pred})

	batch_size = 8
	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
	model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
			steps_per_epoch=max(1, num_train//batch_size),
			validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
			validation_steps=max(1, num_val//batch_size),
			epochs=50,
			initial_epoch=0,
			callbacks=[logging, checkpoint])
	model.save_weights(log_dir + 'trained_weights_stage_1.h5')

	#Fine tuning.
	for i in range(len(model.layers)):
		model.layers[i].trainable = True
	model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
	print('Unfreeze all of the layers.')

	batch_size = 1 # note that more GPU memory is required after unfreezing the body
	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
	model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
		steps_per_epoch=max(1, num_train//batch_size),
		validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
		validation_steps=max(1, num_val//batch_size),
		epochs=100,
		initial_epoch=50,
		callbacks=[logging, checkpoint, reduce_lr, early_stopping])

	model.save_weights(log_dir + 'trained_weights_final.h5')

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
	#Check if the keras model already exists.
	if not os.path.exists(KERAS_MODEL_PATH):
		print("Keras model not found on disk, creating...")
		if not os.path.exists(f"{MODEL_DATA_PATH}/{YOLO_WEIGHTS_FILE_NAME}"):
			download_weights = input("YOLO Weights file is missing, would you like to download it? Y/N: ")
			if download_weights.lower() == "y":
				print("Downloading...")
				get_yolo_weights()
			else:
				print("File not downloaded! Cannot proceed! Exiting...")
				return
			#Read weights and create keras model.
			print("Creating Keras model...")
			convert_to_keras_model()
			print("Done!")
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
		print("Done downloading the images!")
		print("Creating train files...")
		create_train_files(downloaded_images, class_list)
		print("Done creating train files!")
		print("Training the model now...")
		train_model()
		print("Done training! Take a look at the example files on how to use the model.")

if __name__ == "__main__":
	main()