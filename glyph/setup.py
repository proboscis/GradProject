__author__ = 'kentomasui'
import os
import pip
from util import readAppend

PYLEARN2_PATH = '~/pylearn2_data'
PYLEARN2_PATH_ENV = 'export PYLEARN2_DATA_PATH=' + PYLEARN2_PATH




def ensureLineExist(filename, line):
    line += '\n'

    def addLineIfNone(fin, fout):
        #it seems that each line contains new line keyword.
        sum = reduce(lambda a, b: a + b, fin)
        if not line in sum:
            fout.write(line)

    readAppend(filename, addLineIfNone)


if __name__ == '__main__':
    for item in {'cython','theano','matplotlib','ipython'}:
        pip.main(['install',item])
    os.system('python -m ensurepip --upgrade')
    os.chdir('../')
    os.system('git clone git://github.com/lisa-lab/pylearn2.git')
    os.chdir('pylearn2')
    os.system('python setup.py develop')
    if not os.path.exists(PYLEARN2_PATH):
        os.makedirs(name=PYLEARN2_PATH)
    scriptPath = os.getcwd() + '/pylearn2/scripts'
    cmd = 'PATH=$PATH:' + scriptPath + '\nexport PATH'

    def ensureprf(line):
        ensureLineExist('~/.bash_profile', line)

    lines = cmd, PYLEARN2_PATH_ENV
    for l in lines:
        ensureprf(l)
        # download and install XCode!!!


