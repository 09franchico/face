import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import os

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App - Photo and Video")
        self.root.geometry("800x600")

        # Label para exibir o vídeo ao vivo ou a imagem capturada
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(pady=20)
        
        # Botões para tirar a foto e iniciar a detecção de rostos
        self.btn_take_photo = tk.Button(self.root, text="Take Photo", command=self.take_photo)
        self.btn_take_photo.pack(pady=10)
        
        self.btn_start_video = tk.Button(self.root, text="Start Video and Detect Faces", command=self.start_video, state=tk.DISABLED)
        self.btn_start_video.pack(pady=10)

        # Variáveis para capturar o vídeo e armazenar a foto tirada
        self.cap = None
        self.photo_encoding = None
        self.photo_path = None

    def take_photo(self):
        # Abre a webcam e captura a foto
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        ret, frame = self.cap.read()
        if ret:
            # Salva a foto
            if not os.path.exists("captured_photos"):
                os.makedirs("captured_photos")

            self.photo_path = f"captured_photos/photo.png"
            cv2.imwrite(self.photo_path, frame)
            messagebox.showinfo("Photo Captured", f"Photo saved as {self.photo_path}")
            
            # Exibe a foto tirada
            self.show_image(self.photo_path)

            # Faz a codificação do rosto capturado
            image = face_recognition.load_image_file(self.photo_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                print(face_encodings[0])
                self.photo_encoding = face_encodings[0]
                messagebox.showinfo("Success", "Face encoding captured from photo!")
                self.btn_start_video.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "No faces found in the captured photo!")
        else:
            messagebox.showerror("Error", "Failed to capture photo")
        
        self.cap.release()

    def show_image(self, img_path):
        # Carrega e exibe uma imagem na interface
        image = Image.open(img_path)
        image = image.resize((400, 300), Image.LANCZOS)  # Substitua ANTIALIAS por LANCZOS
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

    def start_video(self):
        # Inicia a detecção de rostos usando o vídeo ao vivo
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        self.detect_faces_in_video()

    def detect_faces_in_video(self):
        # Detecta rostos no vídeo ao vivo e compara com o rosto capturado
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([self.photo_encoding], face_encoding)
                distance_face = face_recognition.face_distance([self.photo_encoding], face_encoding)

                index = np.argmin(distance_face)
                nomes = ['Francisco','Flavica']
                if matches[index]:
                    print(nomes[index])


                if True in matches:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Francisco", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "No", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Converte o quadro para exibir no Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

            # Continuar a exibição do vídeo
            self.video_frame.after(10, self.detect_faces_in_video)
        else:
            messagebox.showerror("Error", "Failed to read frame from webcam")
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
