import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk
import os
import pickle

class FacialRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Reconhecimento Facial")
        
        self.video_capture = cv2.VideoCapture(0)
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        self.btn_capture = tk.Button(window, text="Capturar e Gerar Encoding", command=self.capture_and_encode)
        self.btn_capture.pack(side=tk.LEFT, padx=10)
        
        self.btn_detect = tk.Button(window, text="Detectar Faces", command=self.detect_faces)
        self.btn_detect.pack(side=tk.RIGHT, padx=10)
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.load_known_faces()
        
        self.update()
    
    def update(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(15, self.update)
    
    def capture_and_encode(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                cv2.imwrite("captured_face.jpg", face_image)
                
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                
                # Abrir modal para inserir o nome
                name = self.get_person_name()
                if name:
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    
                    self.save_known_faces()
                    
                    messagebox.showinfo("Sucesso", f"Rosto capturado e encoding gerado para {name}")
                else:
                    messagebox.showinfo("Informação", "Operação cancelada pelo usuário")
            else:
                messagebox.showwarning("Aviso", "Nenhum rosto detectado")
    
    def get_person_name(self):
        return simpledialog.askstring("Nome da Pessoa", "Digite o nome da pessoa capturada:")
    
    def detect_faces(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Não encontrada"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Faces Detectadas", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def load_known_faces(self):
        if os.path.exists("known_faces.pkl"):
            with open("known_faces.pkl", "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
    
    def save_known_faces(self):
        with open("known_faces.pkl", "wb") as f:
            pickle.dump({"encodings": self.known_face_encodings, "names": self.known_face_names}, f)
    
    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()