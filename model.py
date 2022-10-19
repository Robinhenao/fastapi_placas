import cv2 
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow import keras as  k


etiquetas2=np.load('etiquetasD.npy','',allow_pickle=True)
dictEtiquetas = etiquetas2.tolist()

to_res = (128, 128,3)
res_model = k.applications.ResNet50(include_top=False, weights="imagenet",input_shape=(128, 128,3))
model = k.models.Sequential()
model.add(k.layers.Lambda(lambda image: tf.image.resize(image, to_res))) 
model.add(res_model)
model.add(k.layers.Flatten())
model.add(k.layers.BatchNormalization())
model.add(k.layers.Dense(256, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.BatchNormalization())
model.add(k.layers.Dense(128, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.BatchNormalization())
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.BatchNormalization())
model.add(k.layers.Dense(37, activation='softmax'))

model = tf.keras.models.load_model('Model_Resnet50_corales.h5')
model.compile(loss='categorical_crossentropy',optimizer=k.optimizers.RMSprop(lr=2e-5),metrics=['accuracy'])


def convertImage(image):
    s = (128,128)
    imagenBlanca = np.ones(s, dtype=np.uint8)*255
    gray2,file1= cv2.threshold(image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w= file1.shape[:2]
    h_img, w_img = imagenBlanca.shape[:2]
    center_x = int(w_img/2)
    center_y = int(h_img/2)
    top_y = center_y - int(h/2)
    left_x = center_x - int(w/2)
    bottom_y = top_y + h
    right_x = left_x + w
    roi = imagenBlanca[top_y:bottom_y, left_x:right_x]
    result = cv2.addWeighted(file1, 1, w, 0, 0)
    imagenBlanca[top_y:bottom_y, left_x:right_x] = result
    imagenBlanca = cv2.cvtColor(imagenBlanca, cv2.COLOR_GRAY2RGB)
    return imagenBlanca

def ocr(imagen,version):
    text= ""
    indicePredicho=0
    
    if (version==1):
        text = pytesseract.image_to_string(imagen, lang='eng', config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890')
    else:
        if (version==2):
            background=[]
            background = convertImage(imagen)
            indicePredicho = (model.predict( np.array( [background,] )  )).argmax(axis=1)
            text = dictEtiquetas[int(indicePredicho)]
    return text

def preprocesamientoModel(ruta):
    file1=cv2.imread(ruta)
    h, s, v = cv2.split(file1)
    v = cv2.subtract(v, 50)
    v[v < 0] = 0
    hsv = cv2.merge((h, s, v))
    img_np = np.array(hsv)
    img_np = file1
    nB = np.matrix(img_np[:,:,0])
    nG = np.matrix(img_np[:,:,1])
    nR = np.matrix(img_np[:,:,2])
    color = cv2.absdiff(nG,nB) #Restar
    imgSize = np.shape(color)
    blockSize = int(1 / 8 * imgSize[0] / 2 * 2 + 1)
    if blockSize <= 1:
        blockSize = int(imgSize[0] / 2 * 2 + 1)
    const = 10
    mask = cv2.adaptiveThreshold(color, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    [fil,col] = color.shape
    for i in range(0,fil):
        for j in range(0,col):
            if color[i,j]<80:
                color[i,j]=0

    for i in range(0,fil):
        for j in range(0,col):
            if color[i,j]>0:
                color[i,j]=1

    color = color *255
    im2,contornos,hierarchy = cv2.findContours(color,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        epsilon = 0.09*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)   #approximate the shape of polygonal curves
        if area>759 and area<140000:     # >1000 and <140000:
            aspect_ratio = float(w)/h
            if aspect_ratio>1.5 and aspect_ratio<10.3:  #original: 2.07 >1.9
                placa = file1[y:y+h,x:x+w]
                candidatos.append(c)
        
    return placa

def segmentadoModel(file2,version):
    dimensions = (162,78)
    image = cv2.resize(file2,dimensions)
    cnts = []
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Actual:
    gray2 = cv2.blur(gray1,(3,3))
    canny1 = cv2.Canny(gray2,150,200)
    canny2 = cv2.dilate(canny1,None,iterations=1)
    #establecer un umbral
    umbral = 100
    ret,umbral_img = cv2.threshold(gray1, umbral, 255, cv2.THRESH_BINARY)
    im2, cnts, jerarquia = cv2.findContours(umbral_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i=1
    lista = list()
    for i in range(len(cnts)):
        placa = []
        area = cv2.contourArea(cnts[i])
        x,y,w,h = cv2.boundingRect(cnts[i])
        epsilon = 0.09*cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],epsilon,True)   #approximate the shape of polygonal curves
        if area>300 and area<1000:   #551-4900      #251-4900  #251-1000    #151 - 1000
            aspect_ratio = float(w)/h
            if aspect_ratio>0.1 and aspect_ratio<2.6:  #original: 2.4 ; >0.1 <10 <2.6
                placa = gray2[y:y+h,x:x+w]
                #text = ""  #No pertenece al código original
                if placa is None:
                    continue  
                text = ocr(placa,version)
                lista.append(str(text))
                print('Caracter Identificado: ',text)
    text="".join(lista)
    return text


