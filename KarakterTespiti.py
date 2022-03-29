
import os

import cv2
import numpy as np
import math
import random

import Main
import Hazirliklar
import PossibleChar

kNearest = cv2.ml.KNearest_create()

MİN_PİKSEL_GENİŞLİĞİ = 2
MİN_PİKSEL_YÜKSEKLİĞİ = 8

MİN_GÖRÜNÜŞ_ORANI = 0.25
MAX_GÖRÜNÜŞ_ORANI = 1.0

MİN_PİKSEL_ALANI = 80

# İki karakteri karşılaştırmak için gerekli sabitler
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

ALANDA_MAKSİMUM_DEĞİŞİKLİK = 0.5

GENİŞLİKTE_MAX_DEĞİŞİM = 0.8
YÜKSEKLİKTE_MAKSİMUM_DEĞİŞİM = 0.2

KARAKTERLER_ARASINDAKİ_MAX_AÇI = 12.0

# Diğer sabitler
MİN_EŞLEŞEN_KARAKTER_SAYISI = 3

YENİDEN_BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_GENİŞLİĞİ = 20
BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_YÜKSEKLİĞİ = 30

MIN_KONTUR_ALANI = 100

def veriYükleÖğrenKNN():
    allContoursWithData = []
    validContoursWithData = []
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except: # Dosya açılmazsa
        print("Hata!,classifications.txt açılamıyor, Programdan çıkılıyor\n")
        os.system("Duraklat")
        return False
    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except: # Dosya açılmazsa
        print("Hata!,flattened_images.txt açılamıyor, Programdan çıkılıyor\n")
        os.system("Duraklat")
        return False

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest.setDefaultK(1)  # Varsayılan K yı 1 e ayarla
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    return True

def plakadanKarakterTespitEt(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []
    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates
    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Hazirliklar.processPreparation(possiblePlate.imgPlate)
        if Main.adimlariGöster == True:
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # Plakayı daha kolay görüntülemek ve tespit etmek için plaka görüntüsünün boyutunu artırıyoruz
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if Main.adimlariGöster == True:
            cv2.imshow("5d", possiblePlate.imgThresh)
         # Plakadaki tüm olası karakterleri bul
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        if Main.adimlariGöster == True:
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]  # Kontur listesini temizle
            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            cv2.drawContours(imgContours, contours, -1, Main.beyaz)
            cv2.imshow("6", imgContours)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if Main.adimlariGöster == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]
            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)
                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            cv2.imshow("7", imgContours)
        if (len(listOfListsOfMatchingCharsInPlate) == 0): # Plakada eşleşen karakter grupları yoksa
            if Main.adimlariGöster == True:
                print("plaka numarasında karakter bulundu " + str(
                    intPlateCounter) + " = (none),Devam etmek için bir resme tıklayın ve bir tuşa basın")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            possiblePlate.strChars = ""
            continue # For döngüsünün başına geri dön
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)  # Karakterleri soldan sağa sırala
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])
        if Main.adimlariGöster == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)
                del contours[:]
                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            cv2.imshow("8", imgContours)
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
        # Plakadaki en uzun eşleşen karakter listesinin gerçek karakter listesi olduğunu varsayalım
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        if Main.adimlariGöster == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]
            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            cv2.drawContours(imgContours, contours, -1, Main.beyaz)
            cv2.imshow("9", imgContours)
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
        if Main.adimlariGöster == True:
            print("Plaka numarasında karakter bulundu " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ",Devam etmek için bir resme tıklayın ve bir tuşa basın")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
    if Main.adimlariGöster == True:
        print("\nKarakter algılama tamamlandı, devam etmek için herhangi bir tuşa basın ve resme tıklayın \n")
        cv2.waitKey(0)
    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    contours = []
    imgThreshCopy = imgThresh.copy()

    # Plakadaki tüm konturları bul
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = PossibleChar.PossibleChar(contour)
        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)  # Olası karakter listesine ekle
    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
        # Bu işlev, bir karakter olup olmadığını görmek için bir kontur üzerinde kaba bir kontrol yapan bir 'ilk geçiştir'
    if (possibleChar.intBoundingRectArea > MİN_PİKSEL_ALANI and
        possibleChar.intBoundingRectWidth > MİN_PİKSEL_GENİŞLİĞİ and possibleChar.intBoundingRectHeight > MİN_PİKSEL_YÜKSEKLİĞİ and
        MİN_GÖRÜNÜŞ_ORANI < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_GÖRÜNÜŞ_ORANI):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfPossibleChars):
     # Bu işlevin amacı, tek büyük karakter listesini eşleşen karakter listelerinin bir listesi halinde yeniden düzenlemektir.
    listOfListsOfMatchingChars = []
    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars) # Büyük listedeki mevcut karakterle eşleşen tüm karakterleri bul
        listOfMatchingChars.append(possibleChar)  # Geçerli karakteri mevcut olası eşleşen karakter listesine ekle
        if len(listOfMatchingChars) < MİN_EŞLEŞEN_KARAKTER_SAYISI: # Mevcut olası eşleşen karakter listesi, olası bir plakayı oluşturmaya yetecek kadar uzun değilse
            continue
        # Olası bir plaka olmak için yeterli karaktere sahip olmadığından listeyi herhangi bir şekilde kaydetmek için
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars: # Özyinelemeli çağrı tarafından bulunan her eşleşen karakter listesi için
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)      # Eşleşen karakterleri orijinal listeye ekle
        break
    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []
    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar: # Eşleşmeleri bulmaya çalıştığımız karakter, şu anda kontrol ettiğimiz büyük listedeki karakterle tam olarak aynı ise
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
        # Karakterlerin eşleşip eşleşmediğini kontrol et
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < KARAKTERLER_ARASINDAKİ_MAX_AÇI and
            fltChangeInArea < ALANDA_MAKSİMUM_DEĞİŞİKLİK and
            fltChangeInWidth < GENİŞLİKTE_MAX_DEĞİŞİM and
            fltChangeInHeight < YÜKSEKLİKTE_MAKSİMUM_DEĞİŞİM):
            listOfMatchingChars.append(possibleMatchingChar)  # Karakterler eşleşiyorsa, mevcut karakteri eşleşen karakterlerin listesine ekle
    return listOfMatchingChars
# İki karakter arasındaki mesafeyi hesaplamak için Pisagor teoremi kullanıldı
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))
# Karakterler arasındaki açıyı hesaplamak için temel trigonometri kullanıldı
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi) # Açıyı derece cinsinden hesapla
    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar: # Mevcut karakter ve diğer karakter aynı karakter değilse
                # Mevcut karakter ve diğer karakter hemen hemen aynı konumda merkez noktalarına sahipse
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea: # Mevcut karakter diğer karakterden daha küçükse
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved: # Mevcut karakter önceki bir geçişte zaten kaldırılmadıysa
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # Sonra mevcut karakteri kaldır
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)
    return listOfMatchingCharsWithInnerCharRemoved
# Gerçek karakter tanımayı uyguladığımız kısım
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX) # Karakterleri soldan sağa sırala
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor) # Eşik görüntüsünün renkli versiyonunu yap, böylece üzerine renkli konturlar çizebiliriz
    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))
        cv2.rectangle(imgThreshColor, pt1, pt2, Main.yesil, 2)  # Karakterin etrafına yeşil kutu çiz
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (YENİDEN_BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_GENİŞLİĞİ, BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_YÜKSEKLİĞİ)) # Görüntüyü yeniden boyutlandır
        npaROIResized = imgROIResized.reshape((1, YENİDEN_BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_GENİŞLİĞİ * BOYUTLANDIRILMIŞ_KARAKTERİN_GÖRÜNTÜ_YÜKSEKLİĞİ))
        npaROIResized = np.float32(npaROIResized) # 1d numpy dizi dizisinden 1d numpy float dizisine dönüştür
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        strChars = strChars + strCurrentChar
    if Main.adimlariGöster == True:
        cv2.imshow("10", imgThreshColor)
    return strChars








