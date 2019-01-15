from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

import os
import time
import random
import math

seed = int (time.time() * 1.e12) % int(time.time())
random.seed(seed)

# Replace with a valid key
training_key = "f64c4178e8e043bda59fc902f9bbf1e4"
prediction_key = "562560d9f2cc442aad31fa46aead14c3"

trainer = training_api.TrainingApi(training_key)
predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

project = trainer.get_project("86e84489-2046-49d9-ba2a-aa9df75564fe")
# Create a new project
#project = trainer.create_project("mjMentorInc", None, 'General', 'Multiclass')

label_data = {}
for label in range(0, 10):
	label_data[str(label)] = ""
label_names = label_data.keys()
	
tags = trainer.get_tags(project.id)
for tag in tags:
	if (tag.name not in label_names):
		trainer.delete_tag(project.id, tag.id)
	else:		
		label_data[tag.name] = tag.id

for label in label_names:
	if (label_data[label] == ""):
		new_tag = trainer.create_tag(project.id, label)
		label_data[label] = new_tag.id
		
		
print("Start training a classifier incrementally...")


root = "trImages"
all_train_images = {}
for label in label_names:
	all_train_images[label] = os.listdir(os.path.join(root, label))

training_samples = 5

# build an initial classifier
if (trainer.get_tagged_image_count(project.id) < training_samples * len(tags)):
	for label in all_train_images.keys():
		images = all_train_images[label]
		random.shuffle(images)
		for i in range(training_samples):
			filepath = os.path.join(root, label, images[i])
			with open(filepath, mode="rb") as img_data: 
				trainer.create_images_from_data(project.id, img_data, [label_data[label]])

	print ("Training...")	
	iteration = trainer.train_project(project.id)
	while (iteration.status != "Completed"):
		iteration = trainer.get_iteration(project.id, iteration.id)
		print ("Training status: " + iteration.status)
		time.sleep(1)

	# The iteration is now trained. Make it the default project endpoint
	trainer.update_iteration(project.id, iteration.id, is_default=True)
	print ("Done!")


run, max_runs = 1, 10
while(run < max_runs):
	correct, number = 0, 0
	for label in all_train_images.keys():
		images = all_train_images[label]
		for i in range(int(random.random() * 1e3) % 5 + 1):
			random.shuffle(images)
		incorrect, total, max_incorrect = 0, 0, 2
		for name in images:
			tag, score = None, 0
			with open(os.path.join(root, label, name), mode="rb") as img_data:
				results = predictor.predict_image(project.id, img_data)

				for prediction in results.predictions:
					if (prediction.probability > score):
						tag, score = prediction.tag_name, prediction.probability
				
			if (tag == label):
				correct = correct + 1
			else:
				incorrect = incorrect + 1
				with open(os.path.join(root, label, name), mode="rb") as img_data:
					trainer.create_images_from_data(project.id, img_data, [label_data[label]])
					
			total = total + 1
			number = number + 1
					
			if (incorrect >= max_incorrect or total >= 2 * training_samples):
				break
				
		acc = 100.0 - float(incorrect) / float(total) * 100.0
		print ("\t [" + label + "] Accuracy: {0:.2f}%".format(acc) + " (" + str(total-incorrect) + "/" + str(total) + ")")
				
	accuracy = float(correct) / float(number) * 100.0
	print ("[Run " + str(run) + "] Accuracy: {0:.2f}%".format(accuracy) + " (" + str(correct) + "/" + str(number) + ")")
	
	if (accuracy < 99):
		# we remove the oldest and also not the default iteration if there are more than 5
		all_iterations = trainer.get_iterations(project.id)
		if len(all_iterations) > 5:
			it_to_be_deleted = None
			for it in all_iterations:
				if it.is_default:
					continue
				if it_to_be_deleted == None or it_to_be_deleted.last_modified > it.last_modified:
					it_to_be_deleted = it
			if it_to_be_deleted is not None: 
				print ("we deleted" + it_to_be_deleted.name)
				trainer.delete_iteration(project.id, it_to_be_deleted.id)
			
		print ("Train and create a new iteration...")	
		iteration = trainer.train_project(project.id)
		while (iteration.status != "Completed"):
			iteration = trainer.get_iteration(project.id, iteration.id)
			print ("Training status: " + iteration.status)
			time.sleep(1)

		# The iteration is now trained. Make it the default project endpoint
		trainer.update_iteration(project.id, iteration.id, is_default=True)
		print ("Done!")
		run = run + 1
	else:
		run = max_runs
