from mlnumbers import classifyNumbers, storeNumbers
from mlmodel import trainModel, checkModel



API_KEY = "dfa359c0-8293-11eb-ac6e-8da14472a72855cd2fb7-6bd2-42d2-8254-624e10426351"


# -------------------------------------------------------
# CHECK IF THE MACHINE LEARNING MODEL IS READY TO USE
# -------------------------------------------------------

# you can use this to check if your machine learning model
# has finished training 

status = checkModel(API_KEY)
print (status)




# -------------------------------------------------------
# USE YOUR MACHINE LEARNING MODEL TO RECOGNIZE NUMBERS 
# -------------------------------------------------------

# CHANGE THIS to the data that you want your 
# machine learning model to classify
data1 = "very well"
data2 = "no"
data3 = 44
data4 = "yes"
data5 = "no"
data6 = "no"
data7 = "yes"
data8 = "yes"
data9 = "yes"
data10 = 91

test_data = [ data1, data2, data3, data4, data5, data6, data7, data8, data9, data10 ]

demo = classifyNumbers(API_KEY, test_data)

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))




# -------------------------------------------------------
# ADD TRAINING EXAMPLES TO YOUR MACHINE LEARNING PROJECT
# -------------------------------------------------------

# CHANGE THIS to the data that you want to add 
# to your project training data
data1 = "very well"
data2 = "yes"
data3 = 0
data4 = "yes"
data5 = "yes"
data6 = "yes"
data7 = "yes"
data8 = "yes"
data9 = "yes"
data10 = 0

training_data = [ data1, data2, data3, data4, data5, data6, data7, data8, data9, data10 ]

# CHANGE THIS to the training bucket to add the
# training example to
training_label = "You_have_to_make_a_test"

# remove the comment on the next line to use this 
storeNumbers(API_KEY, training_data, training_label)




# -------------------------------------------------------
# TRAIN A NEW MACHINE LEARNING MODEL
# -------------------------------------------------------

# after collecting new training examples, you can use 
# to train a new machine learning model 

# remove the comment on the next line to use this 
# trainModel(API_KEY)
