from function.keyWordExtract import keyWord,WordSegmentation






if __name__=="__main__":

    #关键词提取
    keyWord("./otherFunction/doc/01.txt")

    #句法分析
    wordSeg = WordSegmentation(stop_words_file="./textrank4zh/stopwords.txt")
    segment = wordSeg.segmentPos("情况1：仅密码模块丢失、失控或被敌获取时，应及时控制和收回对应装备的IC卡，并转入后续处置流程；")
    print(segment)


    #去除停止词
    segment = wordSeg.segment("情况1：仅密码模块丢失、失控或被敌获取时，应及时控制和收回对应装备的IC卡，并转入后续处置流程；")
    print(segment)






