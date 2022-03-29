
import cv2
import numpy as np
import math

GAUSSIAN_PÜRÜZSÜZ_FİLTRE_BOYUTU = (5, 5)
UYARLANABİLİR_EŞİK_BLOK_BOYUTU = 19
UYARLANABİLİR_EŞİK_AĞIRLIĞI = 9

def processPreparation(orjinalresim):
    rsmGriTon = extractValue(orjinalresim)
    imgMaxContrastGrayscale = maximizeContrast(rsmGriTon)
    height, width = rsmGriTon.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_PÜRÜZSÜZ_FİLTRE_BOYUTU, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, UYARLANABİLİR_EŞİK_BLOK_BOYUTU, UYARLANABİLİR_EŞİK_AĞIRLIĞI)
    return rsmGriTon, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue

def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat











