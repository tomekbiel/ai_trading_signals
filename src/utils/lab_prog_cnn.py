# Importowanie niezbędnych bibliotek
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Wczytanie i przygotowanie danych
def load_and_prepare_data():
    """
    Funkcja ładująca i przygotowująca zbiór danych CIFAR-10.
    Zbiór zawiera 60 000 kolorowych obrazków 32x32 piksele w 10 klasach.
    """
    # Wczytanie danych CIFAR-10, które są już podzielone na zbiór treningowy i testowy
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalizacja wartości pikseli do zakresu [0, 1] poprzez podzielenie przez 255
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Etykiety klas w zbiorze CIFAR-10
    class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń',
                  'pies', 'żaba', 'koń', 'statek', 'ciężarówka']
    
    return train_images, train_labels, test_images, test_labels, class_names

# Wczytanie i przygotowanie danych
train_images, train_labels, test_images, test_labels, class_names = load_and_prepare_data()

# Wizualizacja przykładowych obrazków ze zbioru treningowego
def visualize_sample_images(images, labels, class_names, num_images=25):
    """
    Funkcja wyświetlająca przykładowe obrazy ze zbioru danych.
    
    Parametry:
    - images: tablica z obrazkami
    - labels: etykiety obrazków
    - class_names: lista nazw klas
    - num_images: liczba obrazków do wyświetlenia (domyślnie 25)
    """
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        # Etykiety są zapisane jako tablice 1-elementowe, stąd [0] na końcu
        plt.xlabel(class_names[labels[i][0]])
    plt.tight_layout()
    plt.show()

# Wywołanie funkcji do wizualizacji przykładowych obrazków
print("Przykładowe obrazy ze zbioru treningowego:")
visualize_sample_images(train_images, train_labels, class_names)

def build_cnn_model():
    """
    Funkcja budująca model konwolucyjnej sieci neuronowej (CNN).
    
    Architektura modelu:
    1. Warstwy konwolucyjne (Conv2D) - wyodrębniają cechy z obrazów
    2. Warstwy max-pooling (MaxPooling2D) - zmniejszają wymiary przestrzenne
    3. Warstwa spłaszczająca (Flatten) - przekształca dane do postaci 1D
    4. Gęste warstwy (Dense) - pełne połączenia do klasyfikacji
    
    Wnioski z pierwszego model.summary():
    - Każda warstwa konwolucyjna zwiększa liczbę filtrów (32 -> 64 -> 64)
    - Warstwy max-pooling zmniejszają wymiary o połowę
    - Ostatecznie otrzymujemy tensor 4x4x64 = 1024 wartości
    
    Wnioski z drugiego model.summary() po dodaniu warstw gęstych:
    - Warstwa Flatten przekształca dane do 1024 elementów
    - Warstwa Dense(64) redukuje wymiar do 64 neuronów
    - Warstwa wyjściowa Dense(10) daje wynik dla każdej z 10 klas
    """
    model = models.Sequential()
    
    # Pierwsza warstwa konwolucyjna
    # 32 filtry 3x3, aktywacja ReLU, wejście 32x32x3 (szerokość x wysokość x kanały RGB)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # Warstwa max-pooling 2x2 - zmniejsza wymiary o połowę
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Druga warstwa konwolucyjna z większą liczbą filtrów
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Trzecia warstwa konwolucyjna
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    print("\nArchitektura modelu po warstwach konwolucyjnych:")
    model.summary()
    
    # Spłaszczenie danych do wektora 1D
    model.add(layers.Flatten())
    # W pełni połączona warstwa ukryta z 64 neuronami
    model.add(layers.Dense(64, activation='relu'))
    # Warstwa wyjściowa z 10 neuronami (po jednym na każdą klasę)
    model.add(layers.Dense(10))  # Brak funkcji aktywacji, bo użyjemy from_logits=True w funkcji straty
    
    print("\nPełna architektura modelu:")
    model.summary()
    
    return model

# Budowa modelu
print("\nBudowa modelu CNN...")
model = build_cnn_model()

def compile_and_train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    """
    Funkcja kompilująca i trenująca model.
    
    Parametry kompilacji:
    - optimizer: Adam - adaptacyjny optymalizator gradientowy
    - loss: SparseCategoricalCrossentropy - funkcja straty dla klasyfikacji wieloklasowej
    - metrics: accuracy - dokładność jako metryka ewaluacji
    
    Parametry trenowania:
    - epochs: liczba epok treningowych
    - validation_data: zbiór walidacyjny do oceny modelu po każdej epoce
    """
    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("\nRozpoczęcie trenowania modelu...")
    # Trenowanie modelu
    history = model.fit(train_images, train_labels, 
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        verbose=1)
    
    return history

# Kompilacja i trenowanie modelu
history = compile_and_train_model(model, train_images, train_labels, test_images, test_labels, epochs=10)

def plot_training_history(history):
    """
    Funkcja rysująca wykresy dokładności (accuracy) i straty (loss) w trakcie trenowania.
    
    Parametry:
    - history: obiekt historii zwracany przez model.fit()
    """
    # Wykres dokładności
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Dokładność trenowania')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata trenowania')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_images, test_labels):
    """
    Funkcja oceniająca model na zbiorze testowym.
    """
    print("\nOcena modelu na zbiorze testowym...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Dokładność na zbiorze testowym: {test_acc:.4f}')
    print(f'Strata na zbiorze testowym: {test_loss:.4f}')
    return test_loss, test_acc

# Wizualizacja wyników trenowania
print("\nWizualizacja wyników trenowania:")
plot_training_history(history)

# Ocena modelu na zbiorze testowym
test_loss, test_acc = evaluate_model(model, test_images, test_labels)

# Wnioski
print("\nWNIOSKI:")
print("1. Architektura modelu:")
print("   - 3 warstwy konwolucyjne z rosnącą liczbą filtrów (32 -> 64 -> 64)")
print("   - 2 warstwy max-pooling 2x2 do redukcji wymiarów")
print("   - 1 warstwa gęsta z 64 neuronami i aktywacją ReLU")
print("   - Warstwa wyjściowa z 10 neuronami (bez aktywacji)")
print("\n2. Wyniki:")
print(f"   - Osiągnięta dokładność na zbiorze testowym: {test_acc*100:.2f}%")
print("   - Model uczy się poprawnie, ale istnieje ryzyko przeuczenia (overfitting)")
print("     widoczne po rozbieżności między dokładnością trenowania a walidacji")
print("\n3. Sugestie ulepszeń:")
print("   - Dodanie warstw Dropout do redukcji przeuczenia")
print("   - Zastosowanie BatchNormalization dla lepszej stabilizacji uczenia")
print("   - Zwiększenie liczby warstw lub neuronów w warstwach gęstych")
print("   - Zastosowanie augmentacji danych (obroty, przesunięcia, odbicia)")
print("   - Zwiększenie liczby epok z wczesnym zatrzymaniem (early stopping)")

# Dodatkowa funkcjonalność - predykcja na przykładowym obrazku
def predict_sample(model, images, labels, class_names, num_samples=5):
    """
    Funkcja pokazująca predykcje modelu na przykładowych obrazkach.
    """
    print("\nPrzykładowe predykcje modelu:")
    predictions = model.predict(images[:num_samples])
    
    # Konwersja logitów na prawdopodobieństwa używając funkcji softmax
    predicted_probs = tf.nn.softmax(predictions).numpy()
    
    plt.figure(figsize=(15, 3*num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(images[i])
        true_label = class_names[labels[i][0]]
        predicted_label = class_names[tf.argmax(predicted_probs[i])]
        color = 'green' if true_label == predicted_label else 'red'
        plt.title(f'Prawdziwa: {true_label}\nPrzewidziana: {predicted_label}', color=color)
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.bar(range(10), predicted_probs[i])
        plt.xticks(range(10), class_names, rotation=90)
        plt.title('Rozkład prawdopodobieństwa')
    
    plt.tight_layout()
    plt.show()

# Wywołanie funkcji do pokazania przykładowych predykcji
predict_sample(model, test_images, test_labels, class_names, num_samples=5)