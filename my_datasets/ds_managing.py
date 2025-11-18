import os
import random
import string
import cv2

# Ścieżka do folderu z plikami .jpg
folder_path = "./my_scans/"

# Funkcja do generowania unikalnych 8-znakowych nazw
def generate_unique_name(existing_names):
    while True:
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if name not in existing_names:
            existing_names.add(name)
            return name

existing_names = set()

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
        full_path = os.path.join(folder_path, filename)
        
        # Wczytaj obraz
        img = cv2.imread(full_path)
        if img is None:
            print(f"Nie udało się wczytać pliku {filename}, pomijam.")
            continue
        
        # Zmień rozmiar na 124x124 px
        img_resized = cv2.GaussianBlur(img, (11, 11), 0)
        img_resized = cv2.resize(img_resized, (124, 124))
        
        # Wygeneruj unikalną nazwę
        new_name = generate_unique_name(existing_names) + ".jpg"
        new_path = os.path.join(folder_path, new_name)
        
        # Zapisz zmieniony obraz
        cv2.imwrite(new_path, img_resized)
        
        # Usuń stary plik
        os.remove(full_path)

print("Zakończono przetwarzanie wszystkich plików.")
