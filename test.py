import pickle
import torch


# Load the pickle file
with open('./adult/adult_autoint_train.pickle', 'rb') as f:
    data = pickle.load(f)

# Print the type of data and some basic information
print("Type of data:", type(data))

# Depending on the structure of your data, you can further inspect it
# For example, if it's a dictionary, you can print keys:
if isinstance(data, dict):
    print("Keys in the dictionary:", data.keys())

# Or you can simply print the data itself to see its structure
print("Data:", data)

predict_model = torch.load("./adult/adult_autoint_model.pth", map_location=torch.device('cuda:0'))
print(predict_model)
