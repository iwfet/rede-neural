import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

file_path = 'caminho_para_o_novo_arquivo_excel.xlsx'
df = pd.read_excel(file_path)
df = pd.get_dummies(df, columns=['Estado'])

encoder = LabelEncoder()
df['Semente'] = encoder.fit_transform(df['Semente'])

X = df.drop('Em kg/ha', axis=1)
y = df['Em kg/ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(34, activation='relu'),
    Dense(34, activation='relu'),
    Dense(34, activation='relu'),
    Dense(24, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

loss = model.evaluate(X_test_scaled, y_test)
print(f'Loss (Erro Quadrático Médio) no conjunto de teste: {loss}')

predictions = model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, predictions)

print(f'Coeficiente de determinação (R²): {r2}')
limite = 0.1  
corretos = sum(abs(predictions - y_test.to_numpy().flatten()) < limite)
total = len(y_test)
accuracy = corretos / total

print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Nível de acertos: {corretos}/{total}')

model.save('meu_modelo.h5')
