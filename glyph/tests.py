__author__ = 'kento'
import util
import cPickle as pickle
def saveTest():
    data = [0,1,2,3,4]
    name = "test.pkl"
    util.save(data,name)
    data2 = util.load(name)

    print data, data2
    return data == data2


if __name__ == '__main__':
    saveTest()