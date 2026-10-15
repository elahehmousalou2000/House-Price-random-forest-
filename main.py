from sklearn.ensemble import RandomForestRegressor

X = [[50, 1, 1], [70, 2, 1], [100, 3, 2], [120, 3, 2], [150, 4, 3]]
y = [100000, 140000, 200000, 240000, 300000]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("Prediction:", model.predict([[80, 2, 1]])[0])
