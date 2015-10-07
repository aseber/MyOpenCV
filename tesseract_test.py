import tesseract
api = tesseract.TessBaseAPI()
api.SetOutputName("outputName");
api.Init(".","eng",tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)
mImgFile = "card.png"
pixImage=tesseract.pixRead(mImgFile)
api.SetImage(pixImage)

# lst = dir(tesseract)

# f = open("output.txt", 'w')

# for line in lst:
    # f.write(line)
    # f.write('\n')

outText=api.GetUTF8Text()
print("OCR output:\n%s"%outText);
api.End()