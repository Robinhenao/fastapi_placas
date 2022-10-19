import os
from fastapi import FastAPI, Form,UploadFile,File,Body,Request
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import pytesseract
import numpy as np
import model


pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract'


app = FastAPI(title='placas',description='convierte placa en texto',version='1.0')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
def load_index(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post('/submitUpload')
async def post_upload_file(request: Request,flexRadioDefault:bool=Form(...), upload_file:UploadFile=File(...)):
    try:
        with open(upload_file.filename,"wb") as buffer:
            content =await upload_file.read()
            buffer.write(content)
            buffer.close()
            os.rename(upload_file.filename,"imagen.jpg")

            if(flexRadioDefault):
                preprocesamiento("imagen.jpg")
                os.remove("imagen.jpg")
                #segmentacion("proces.jpg")
                #print(pytesseract.image_to_string("letras/0 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                #print(pytesseract.image_to_string("letras/1 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                #print(pytesseract.image_to_string("letras/2 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                #print(pytesseract.image_to_string("letras/3 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                #print(pytesseract.image_to_string("letras/4 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                #print(pytesseract.image_to_string("letras/5 - letras.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890'))
                text=pytesseract.image_to_string("proces.jpg",config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIjJKLMNOPQRSTUVWXYZ-1234567890')
                os.remove("proces.jpg")
                #img=model.preprocesamientoModel("imagen.jpg")
                #os.remove("imagen.jpg")
                #text=model.segmentadoModel(img,1)
            else:
                img=model.preprocesamientoModel("imagen.jpg")
                os.remove("imagen.jpg")
                text=model.segmentadoModel(img,2)
                #text="model"     
        return templates.TemplateResponse("show_text.html",{"request":request,"text":text})
    except FileNotFoundError:
        return templates.TemplateResponse("index.html",{"request":request})
    



def preprocesamiento(ruta):
    imagen=cv2.imread(ruta)
    rgB=np.matrix(imagen[:,:,0])
    rGb=np.matrix(imagen[:,:,1])
    Rgb=np.matrix(imagen[:,:,2])
    img=cv2.absdiff(rGb,rgB)
    [fil,col]=img.shape
    for i in range(0,fil):
        for j in range(0,col):
            if img[i,j]<80:
                img[i,j]=0
    for i in range(0,fil):
        for j in range(0,col):
            if img[i,j]>0:
                img[i,j]=1
    se=np.ones((50,50), np.uint8)
    se2=np.ones((10,10),np.uint8)
    closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,se)
    dilation=cv2.dilate(closing, se2,1)
    S,contours,hierarchy=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[:]
    num=len(cnt)
    box=np.zeros((num,4))
    for i in range(0,num):
        box[i,:]=cv2.boundingRect(cnt[i])
        L=np.zeros((num,4))
        Max=[0,0]
        for i in range (0,num):
            L[i,:]=box[i]
            if L[i,2]>Max[1]:
                Max=[i,L[i,2]]
    BOX=box[Max[0],:]
    BOX=np.array(BOX,dtype = np.uint32)
    b=imagen[BOX[1]:BOX[1]+BOX[3],BOX[0]:BOX[0]+BOX[2],:]
    cv2.imwrite("proces.jpg",b)



def segmentacion(rutaImagenRecortada):
    A=cv2.imread(rutaImagenRecortada)
    [fil,col,cap]=A.shape
    rgB=A[:,:,0]
    rGb=A[:,:,1]
    Rgb=A[:,:,2]
    R=Rgb/255.0
    G=rGb/255.0
    B=rgB/255.0
    K=np.zeros((fil,col))
    for i in range(0,fil):
        for j in range(0,col):
            MAX=max(R[i,j],G[i,j],B[i,j])
            K[i,j]=1-MAX
    cv2.imwrite("k.bmp",K)
    k=cv2.imread("k.bmp")
    BW1=cv2.Laplacian(k,cv2.CV_8UC1)
    Image=BW1[:,:,0]+BW1[:,:,1]+BW1[:,:,2]
    ret,thresh=cv2.threshold(Image,0,255,0)
    S,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[:]
    num=len(cnt)
    box=np.zeros((num,4))
    for j in range (0,num):
        box[j,:]=cv2.boundingRect(cnt[j])
    Box=np.zeros((20,4))
    [L,A]=thresh.shape
    q=0
    for j in range(0,num):
        p=box[j,:]
        if p[2]>=0.095*A and p[2]<=0.15*A and p[3]>=0.46*L and p[3]<=0.67*L:
            Box[q]=p
            q=q+1 
    Box=np.array(Box,dtype = np.uint32)
    img=cv2.imread(rutaImagenRecortada)

    scale_percent = 120 
    width = int(img.shape[1] * 60 / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    for i in range(0,6):
        let=img[Box[i*2,1]:Box[i*2,1]+Box[i*2,3],Box[i*2,0]:Box[i*2,0]+Box[i*2,2]]
        resized = cv2.resize(let, dim, interpolation = cv2.INTER_AREA)  
        cv2.imwrite("letras/"+str(i)+" - letras.jpg",resized)
    