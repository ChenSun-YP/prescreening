import pickle

file_path = r"data\crosscorrs_edge_mean_True_semifine (1).pkl"

# "data/analysis/selected_neurons_first_200s/autocorrs_edge_mean_True_ultra-fine.pkl"
# "data/selected_neurons_first_200s.pkl"
# "data\\selected_neurons1150b034.pkl"

# Open the file in binary mode ('rb' for read binary) and load the data
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
# print(data)

print("Top-level type:", type(data))

# if isinstance(data, dict):
#     for k in data:
#         print(k, "->", type(data[k]))
#         try:
#             print("  Sample:", data[k][:5])
#         except:
#             print("  Preview:", str(data[k])[:100])

if hasattr(data, "columns"):
    print("Columns:", data.columns.tolist())
elif isinstance(data, dict):
    print("Keys:", list(data.keys()))
elif isinstance(data, list) and data and isinstance(data[0], dict):
    print("Keys from first entry:", list(data[0].keys()))
else:
    print("Unknown structure")

# printed output of "data\Jan2010-Nonstationarity_Learning\1089u195merge-clean_cutoff_5.pkl"
"""
Top-level type: <class 'dict'>
Keys: ['version', 'comment', 'freq', 'tbeg', 'tend', 'neurons', 'events', 'intervals', 'waves', 'contvars', 'popvectors', 'markers']
"""
