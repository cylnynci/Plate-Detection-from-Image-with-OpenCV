
import cv2
import numpy as np
import os

import KarakterTespiti
import PlakaTespiti
import PossiblePlate

siyah = (0.0, 0.0, 0.0)
beyaz = (255.0, 255.0, 255.0)
sari = (0.0, 255.0, 255.0)
yesil = (0.0, 255.0, 0.0)
kirmizi = (0.0, 0.0, 255.0)
adimlariGöster = False

def main():
    ögrenmeBasarisiKNN = KarakterTespiti.veriYükleÖğrenKNN()
    if ögrenmeBasarisiKNN == False:
        print("\nHata! : KNN Başarılı Bir Şekilde Uygulanamadı\n")
        return
    orjinalResim  = cv2.imread("AracListesi/Resim10.png")
    if orjinalResim is None: # Eğer görüntü başarı ile okunmadıysa
        print("\nHata! : Dosyadan Resim Okunamadı \n\n")
        os.system("Pause")
        return
    ihtimalPlakaListesi = PlakaTespiti.plakaTespitEt(orjinalResim)
    ihtimalPlakaListesi = KarakterTespiti.plakadanKarakterTespitEt(ihtimalPlakaListesi)
    cv2.imshow("Originalİmg", orjinalResim)
    if len(ihtimalPlakaListesi) == 0:
        print("\nResimden Plaka Tespit Edilemedi\n")
    else:
         # Olası plakaların listesini DESCENDING sırasına göre sıralayın (çoğu karakter sayısından en az karakter sayısına kadar)
        ihtimalPlakaListesi.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        Plaka = ihtimalPlakaListesi[0]
        cv2.imshow("imgPlate", Plaka.imgPlate)
        cv2.imshow("imgThresh", Plaka.imgThresh)
        if len(Plaka.strChars) == 0:
            print("\nKarakter Tespit Edilemedi.\n\n")
            return
        plakaCevresineKirmiziDörtgenCiz(orjinalResim, Plaka) # Plakanın etrafına kırmızı dikdörtgen çiz
        print("\nResimden Okunan Plaka = " + Plaka.strChars + "\n") # Stdout'a plaka metnini yaz
        print("----------------------------------------")
        resimePlakalariIsle(orjinalResim, Plaka) # Resmin üzerine plaka metnini yazın
        cv2.imshow("Orjinal Resim", orjinalResim)
        cv2.imwrite("OrjinalResim.jpg", orjinalResim)
    cv2.waitKey(0)
    return

def plakaCevresineKirmiziDörtgenCiz(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), kirmizi, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), kirmizi, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), kirmizi, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), kirmizi, 2)

def resimePlakalariIsle(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0
    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX # Düz bir jane yazı tipi seçin
    fltFontScale = float(plateHeight) / 30.0 # Plaka alanının yüksekliğine göre temel yazı tipi boyutu
    intFontThickness = int(round(fltFontScale * 1.5)) # Yazı tipi ölçeğinin temel yazı tipi kalınlığı
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness) # getTextSize'ı çağır
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)
    ptCenterOfTextAreaX = int(intPlateCenterX)
    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))
    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, sari, intFontThickness)

if __name__ == "__main__":
    main()


















