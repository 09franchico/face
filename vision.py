import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Rosto com SSD")

        # Botão para abrir um vídeo
        self.btn_open_video = tk.Button(self.root, text="Abrir Vídeo", command=self.open_video)
        self.btn_open_video.pack()

        # Botão para usar a webcam
        self.btn_webcam = tk.Button(self.root, text="Usar Webcam", command=self.use_webcam)
        self.btn_webcam.pack()

        # Label para exibir o vídeo
        self.label = tk.Label(self.root)
        self.label.pack()

        # Variável para capturar vídeo
        self.cap = None

        # Carregar o modelo SSD
        self.net = self.load_ssd_model()

    def load_ssd_model(self):
        """Função para carregar o modelo SSD"""
        model_file = "deploy.prototxt.txt"  # Arquivo do modelo
        weights_file = "res10_300x300_ssd_iter_140000.caffemodel"  # Pesos pré-treinados
        net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
        return net

    def detect_faces(self, frame):
        """Função para detectar rostos no frame"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Desenhar um retângulo ao redor do rosto detectado
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Exibir a confiança como texto ao lado do rosto
                text = f"{confidence * 100:.2f}%"  # Exibir a confiança como percentual
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return frame

    def open_video(self):
        """Função para abrir um arquivo de vídeo"""
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.capture_video(file_path)

    def use_webcam(self):
        """Função para usar a webcam"""
        self.capture_video(0)

    def capture_video(self, source=0):
        """Função para capturar vídeo de arquivo ou webcam"""
        if self.cap:
            self.cap.release()  # Libera a captura anterior se houver
        self.cap = cv2.VideoCapture(source)
        self.update_frame()

    def update_frame(self):
        """Atualizar o frame no label"""
        ret, frame = self.cap.read()
        if ret:
            # Detectar rostos no frame
            frame = self.detect_faces(frame)

            # Converter o frame para exibir na interface Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        
        # Atualizar o frame continuamente
        self.label.after(10, self.update_frame)

# Interface gráfica Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
