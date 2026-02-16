import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def run_section(name, section_func):
    print(f"\n{'=' * 50}")
    print(f"Uruchamiam sekcję: {name}")
    print("Naciśnij Enter, aby kontynuować...")
    input()
    section_func()
    print(f"Zakończono sekcję: {name}")
    print(f"{'=' * 50}\n")


def section1():
    print("1. Wczytywanie danych...")
    global train_images, train_labels, test_images, test_labels, class_names
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("Dane załadowane pomyślnie!")


def section2():
    print("2. Wyświetlanie przykładowych obrazów...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


def section3():
    print("3. Budowa modelu CNN...")
    global model

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    print("Model zbudowany pomyślnie!")
    model.summary()


def section4():
    print("4. Kompilacja i trenowanie modelu...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Rozpoczynam trenowanie (może to zająć kilka minut)...")
    global history
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))
    print("Trenowanie zakończone!")


def section5():
    print("5. Ewaluacja i wykresy...")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history['accuracy'], label='Dokładność trenowania')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Dokładność na zbiorze testowym: {test_acc:.4f}')


# Główna pętla programu
if __name__ == "__main__":
    print("Witaj w interaktywnym przewodniku po sieci neuronowej CNN!")
    print("Będziemy krok po kroku budować i trenować model do rozpoznawania obrazów CIFAR-10.\n")

    sections = [
        ("1. Wczytanie danych", section1),
        ("2. Wizualizacja danych", section2),
        ("3. Budowa modelu", section3),
        ("4. Trenowanie modelu", section4),
        ("5. Ewaluacja i wykresy", section5)
    ]

    for name, func in sections:
        run_section(name, func)

    print("Gratulacje! Zakończyłeś cały proces uczenia maszynowego!")
    print("Możesz teraz ponownie uruchomić program, aby przejść przez wszystkie kroki jeszcze raz.")