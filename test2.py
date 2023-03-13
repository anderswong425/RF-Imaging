
# create a list of tuples with data
data = [("apple", 1.25), ("banana", 0.75), ("orange", 0.50), ("pear", 1.75)]

# find the maximum length of the first element in each tuple
max_len = max(len(x[0]) for x in data)

print(max_len)
# iterate over the data and format each tuple as a table row
for item in data:
    name, price = item
    print(f"{name:<{max_len}}  {price:>5.2f}")
