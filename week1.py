import numpy as np 
import pandas as pd



one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)


two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)


sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)


random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)


random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1) 


random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)


random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)


# the first Task .
feature = np.arange(6, 21)
print(feature)
label = (feature * 3) + 4
print(label)


# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])

# the second Task 
# Create a 3x4 DataFrame with random integers between 0 and 100
data = np.random.randint(0, 101, size=(3, 4))
columns = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
df = pd.DataFrame(data, columns=columns)

# Output the entire DataFrame
print("Entire DataFrame:")
print(df)

# Output the value in the cell of row #1 of the Eleanor column
eleanor_value_row1 = df.at[1, 'Eleanor']
print("\nThe value in the cell of row #1 of the Eleanor column:", eleanor_value_row1)

# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason
df['Janet'] = df['Tahani'] + df['Jason']

# Output the DataFrame with the Janet column added
print("\nDataFrame with the Janet column:")
print(df)