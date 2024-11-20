import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load CSV data

# Load drivers data
drivers = pd.read_csv('data/drivers.csv')

# Load race results
results = pd.read_csv('data/results.csv')

# Load qualifying data
qualifying = pd.read_csv('data/qualifying.csv')

# Load circuit data
circuits = pd.read_csv('data/circuits.csv')

# Load races data
races = pd.read_csv('data/races.csv')

# Load constructor data
constructors = pd.read_csv('data/constructors.csv')

# Load constructor standings data
constructor_standings = pd.read_csv('data/constructor_standings.csv')

# Load driver standings data
driver_standings = pd.read_csv('data/driver_standings.csv')

'''
# Load free practice data
free_practice = pd.read_csv('data/free_practice.csv')
'''

# Step 2: Data Preprocessing

# Merge results with races to include circuit information
results = pd.merge(results, races[['raceId', 'year', 'round', 'circuitId']], on='raceId')
results = pd.merge(results, circuits[['circuitId', 'name', 'location', 'country']], on='circuitId')
results = pd.merge(results, drivers[['driverId', 'driverRef', 'nationality']], on='driverId')
results = pd.merge(results, constructors[['constructorId', 'name']], on='constructorId', suffixes=('_driver', '_constructor'))

# Merge qualifying data with results for qualifying positions
qualifying = qualifying[['raceId', 'driverId', 'position']]
results = pd.merge(results, qualifying, on=['raceId', 'driverId'], how='left', suffixes=('', '_qualifying'))

# Merge constructor standings to get constructor performance up to the race
results = pd.merge(results, constructor_standings[['raceId', 'constructorId', 'points', 'position']], 
                   on=['raceId', 'constructorId'], how='left', suffixes=('', '_constructor_standing'))

# Step 3: Feature Engineering

# Creates a feature for home race
results['is_home_race'] = results.apply(lambda row: 1 if row['nationality'] == row['country'] else 0, axis=1)

# Average position over the last 5 races
results['recent_form'] = results.groupby('driverId')['positionOrder'].transform(lambda x: x.rolling(window=5).mean())

# Recent constructor points
results['recent_constructor_points'] = results.groupby('constructorId')['points_constructor_standing'].transform(lambda x: x.rolling(window=5).mean())

# Filter columns to relevant features for prediction
X = results[['position_qualifying', 'is_home_race', 'recent_form', 'year', 'recent_constructor_points']]

y = results['positionOrder']  # Assuming 'positionOrder' is the final race position

# Convert categorical variables to numeric
X = pd.get_dummies(X)


# Step 4: Model Training

# Remove rows with any missing values in X
X = X.dropna()

# Correspondingly remove entries in y to match X
y = y[X.index]
#print("X shape after dropping NaNs:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Step 5: Making Predictions with New Data

# Example of new race data input (this should be replaced with actual data)
new_race_data = pd.DataFrame({
    # Race Id for every driver. Can find these in races.csv
    'raceId': [1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141],
    # Driver Ids for each driver organized in qualifying position. Can be found in drivers.csv
    'driver_id': [846, 847, 852, 839, 859, 844, 848, 857, 4, 840, 822, 815, 842, 1, 860, 861, 830, 807, 855, 832],
    # Qualifying positions for every driver. 
    'position_qualifying': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  
    # Checks to see if this is a driver's home race. 0 if not, 1 if so
    'is_home_race': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Recent form. Takes results from the last five races to find average finishing position        
    'recent_form': [2.8, 5, 13, 14.6, 12.5, 2.4, 10.67, 3.8, 9.5, 14.75, 15.8, 10.5, 13.2, 6, 10, 10.6, 4.4, 10.8, 16.2, 3.33], 
    # Indicate year of race   
    'year': [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
    # Sums the constructor's number of points over the past five races      
    'recent_constructor_points': [162, 90, 2, 1, 2, 167, 13, 162, 12, 12, 0, 78, 1, 90, 19, 13, 78, 19, 0, 167]
})

# Same qualifing order as above for printing out predicitons in order
qualifying_order = [846, 847, 852, 839, 859, 844, 848, 857, 4, 840, 822, 815, 842, 1, 860, 861, 830, 807, 855, 832]

# Convert categorical variables to numeric if necessary
new_race_data = pd.get_dummies(new_race_data)

# Align with the training data features
missing_cols = set(X.columns) - set(new_race_data.columns)
for c in missing_cols:
    new_race_data[c] = 0

# Reorder columns to match the training data
new_race_data = new_race_data[X.columns]

# Make predictions with the trained model
predictions = model.predict(new_race_data)
print(f"Predicted race positions: {predictions}")
print()

# Dictionary containing every driver's driverId
driver_dict = {
    830: 'Max Verstappen',
    815: 'Sergio Perez',
    844: 'Charles Leclerc',
    832: 'Carlos Sainz',
    1: 'Lewis Hamilton',
    847: 'George Russell',
    846: 'Lando Norris',
    857: 'Oscar Piastri',
    4: 'Fernando Alonso',
    840: 'Lance Stroll',
    839: 'Esteban Ocon',
    842: 'Pierre Gasly',
    822: 'Valteri Bottas',
    855: 'Zhou Guanyu',
    848: 'Alex Albon',
    861: 'Franco Colapinto',
    825: 'Kevin Magnussen',
    807: 'Nico Hulkenberg',
    859: 'Liam Lawson',
    852: 'Yuki Tsunoda',
    860: 'Oliver Bearman'
}

# Sets position equal to 0
position = 0
# Sets driver equal to the current driver in qualifying order
driver = qualifying_order[position]
# Sets prediction equal to the current prediction
prediction = predictions[position]

# Creates a new list for the updated predictions 
updated_predictions = []

# Loops through every poistion printing out each driver's finishing position 
while (position < 19):
    updated_predictions.append(driver_dict[driver] + ": " + str(prediction))
    position+= 1
    driver = qualifying_order[position]
    prediction = predictions[position]

# Loops through every prediction in updated predictions
for prediction in updated_predictions:
    print(prediction)
    print()
