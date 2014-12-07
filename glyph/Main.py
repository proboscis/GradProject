def sigmoid():
    from theano import tensor as T

    x = T.dmatrix()
    return -1 / (1 + T.exp(-x))

if __name__ == '__main__':
    import theano
    import LRTest
    theano.printing.debugprint(sigmoid())
    theano.printing.pydotprint_variables(sigmoid())
    test = LRTest.LRTest()
    test.doTrain()


def groupLen(seq, chunk_size):
    return list(zip([iter(seq)] * chunk_size))

