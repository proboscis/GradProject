import train,util,json
if __name__ == '__main__':
    import sys
    print "trainTest!"
    #file = "/home/kento/Documents/GradProject/params/ebook_color_sda1.json"
    file = sys.argv[1]
    print "loading file:",file
    jsonLines = util.fileLines(file)
    jsonStr = reduce(lambda a,b: a + b,jsonLines)
    info = json.loads(jsonStr)
    # ebook = train.createEbookDataSet(info["dataSet"]).get_value(borrow=True)
    # mnist = train.createMnistDataSet(info["dataSet"]).get_value(borrow=True)
    #
    # print "========TRAIN========"
    # print "ebook way:",ebook.shape
    # print "mnist way:",mnist.shape
    # def showValues(data):
    #     return reduce(lambda a,b: a+b,data)
    # print "ebook sum:",showValues(ebook[0])
    # print "ebook data:",ebook[0]
    # print "mnist sum:",showValues(mnist[0])
    # print "mnist data:",mnist[0]
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    # plt.imshow(ebook[0].reshape(28,28))
    # plt.figure(1)
    # plt.imshow(mnist[0].reshape(28,28))
    # plt.show()
    # print "========DONE!========"
    train.evalModel(json.loads(jsonStr))
    print "trainTest done!"