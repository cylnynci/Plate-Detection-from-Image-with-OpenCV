
import cv2
import numpy as np
import math
import Main
import random

import Hazirliklar
import KarakterTespiti
import PossiblePlate
import PossibleChar

PLAKA_GENİŞLİK_DOLGU_FAKTÖRÜ = 1.3
PLAKA_YÜKSEKLİĞİ_DOLGU_FAKTÖRÜ = 1.5

def plakaTespitEt(imgOriginalScene):
    listOfPossiblePlates = []
    height, width, numChannels = imgOriginalScene.shape
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    cv2.destroyAllWindows()
    if Main.adimlariGöster == True:
        cv2.imshow("0", imgOriginalScene)
    imgGrayscaleScene, imgThreshScene = Hazirliklar.processPreparation(imgOriginalScene) # Gri tonlamalı ve eşik görüntüleri elde etmek için ön işlem
    if Main.adimlariGöster == True:
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    #bu işlev önce tüm konturları bulur, sonra yalnızca karakter olabilecek konturları içerir (henüz diğer karakterlerle karşılaştırılmadan)
    listOfPossibleCharsInScene = olasıKarakterleriBul(imgThreshScene)
    if Main.adimlariGöster == True:
        print("Adım 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))
        imgContours = np.zeros((height, width, 3), np.uint8)
        contours = []
        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        cv2.drawContours(imgContours, contours, -1, Main.beyaz)
        cv2.imshow("2b", imgContours)
    listOfListsOfMatchingCharsInScene = KarakterTespiti.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    if Main.adimlariGöster == True:
        print("Adım 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))
        imgContours = np.zeros((height, width, 3), np.uint8)
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)
            contours = []
            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        cv2.imshow("3", imgContours)
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)
        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)
    print("\n" + str(len(listOfPossiblePlates)) + " Olası Plakalar Bulundu")
    if Main.adimlariGöster == True:
        print("\n")
        cv2.imshow("4a", imgContours)
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)
            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.kirmizi, 2)
            cv2.imshow("4a", imgContours)
            print("Olası Plaka" + str(i) + ", Devam etmek için bir resme tıklayın ve bir tuşa basın")
            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        print("\nPlaka tespiti tamamlandı, herhangi bir resme tıkla ve bir tuşa tıkla ve karakter tanımayı başlat \n")
        cv2.waitKey(0)
    return listOfPossiblePlates

def olasıKarakterleriBul(imgThresh):
    listOfPossibleChars = []
    intCountOfPossibleChars = 0
    imgThreshCopy = imgThresh.copy()
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Tüm kontürleri bul
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    for i in range(0, len(contours)):
        if Main.adimlariGöster == True:
            cv2.drawContours(imgContours, contours, i, Main.beyaz)
        possibleChar = PossibleChar.PossibleChar(contours[i])
        if KarakterTespiti.checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            listOfPossibleChars.append(possibleChar)
    if Main.adimlariGöster == True:
        print("\nAdım 2 - len(contours) = " + str(len(contours)))
        print("Adım 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))
        cv2.imshow("2a", imgContours)
    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX) # X konumuna göre karakterleri soldan sağa sırala
    # Plakanın merkez noktasını hesapla
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
     # Plaka genişliğini ve yüksekliğini hesapla
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLAKA_GENİŞLİK_DOLGU_FAKTÖRÜ)
    intTotalOfCharHeights = 0
    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    intPlateHeight = int(fltAverageCharHeight * PLAKA_YÜKSEKLİĞİ_DOLGU_FAKTÖRÜ)
    # Plaka bölgesinin düzeltme açısını hesaplayın
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = KarakterTespiti.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    height, width, numChannels = imgOriginal.shape
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height)) # Tüm resmi döndür
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    possiblePlate.imgPlate = imgCropped
    return possiblePlate












