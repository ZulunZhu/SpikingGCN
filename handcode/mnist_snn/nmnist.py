import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class event():
    def __init__(self, xEvent, yEvent, pEvent, tEvent):
        if yEvent is None:
            self.dim = 1
        else:
            self.dim = 2

        self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent)
        self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent)
        self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent)
        self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent)

        if not issubclass(self.x.dtype.type, np.integer): self.x = self.x.astype('int')
        if not issubclass(self.p.dtype.type, np.integer): self.p = self.p.astype('int')

        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer): self.y = self.y.astype('int')

        self.p -= self.p.min()

    def toSpikeArray(self, samplingTime=1, dim=None):
        if self.dim == 1:
            if dim is None: dim = ( np.round(max(self.p)+1).astype(int),
                                    np.round(max(self.x)+1).astype(int),
                                    np.round(max(self.t)/samplingTime+1).astype(int) )
            frame = np.zeros((dim[0], 1, dim[1], dim[2]))
        elif self.dim == 2:
            if dim is None: dim = ( np.round(max(self.p)+1).astype(int),
                                    np.round(max(self.y)+1).astype(int),
                                    np.round(max(self.x)+1).astype(int),
                                    np.round(max(self.t)/samplingTime+1).astype(int) )
            frame = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        return self.toSpikeTensor(frame, samplingTime).reshape(dim)

    def toSpikeTensor(self, emptyTensor, samplingTime=1, randomShift=False):
        if randomShift is True:
            tSt = np.random.randint(max(int(self.t.min() / samplingTime), int(self.t.max() / samplingTime) - emptyTensor.shape[3], emptyTensor.shape[3] - int(self.t.max() / samplingTime), 1))
        else:
            tSt = 0

        xEvent = np.round(self.x).astype(int)
        pEvent = np.round(self.p).astype(int)
        tEvent = np.round(self.t / samplingTime).astype(int) - tSt

        if self.dim == 1:
            validInd = np.argwhere((xEvent < emptyTensor.shape[2]) &
                                (pEvent < emptyTensor.shape[0]) &
                                (tEvent < emptyTensor.shape[3]) &
                                (xEvent >= 0) &
                                (pEvent >= 0) &
                                (tEvent >= 0))
            emptyTensor[pEvent[validInd], 0, xEvent[validInd], tEvent[validInd]] = 1 / samplingTime
        elif self.dim == 2:
            yEvent = np.round(self.y).astype(int)
            validInd = np.argwhere((xEvent < emptyTensor.shape[2]) &
                                (yEvent < emptyTensor.shape[1]) &
                                (pEvent < emptyTensor.shape[0]) &
                                (tEvent < emptyTensor.shape[3]) &
                                (xEvent >= 0) &
                                (yEvent >= 0) &
                                (pEvent >= 0) &
                                (tEvent >= 0))
            emptyTensor[pEvent[validInd], yEvent[validInd], xEvent[validInd], tEvent[validInd]] = 1 / samplingTime
        return emptyTensor

def read2Dspikes(filename):
    if filename[8] == 'r':
        while len(filename) != 22:
            filename = filename[:13] + '0' + filename[13:]
    else:
        while len(filename) != 21:
            filename = filename[:12] + '0' + filename[12:]
    with open(filename, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = np.asarray([x for x in inputByteArray])
    xEvent = inputAsInt[0::5]
    yEvent = inputAsInt[1::5]
    pEvent = inputAsInt[2::5] >> 7
    tEvent = ((inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5])) & 0x7FFFFF
    return event(xEvent+1, yEvent+1, pEvent, tEvent/15000)

class nmnist(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        super(nmnist, self).__init__()
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        inputSpikes = read2Dspikes(self.path + str(inputIndex.item()) + '.bin').toSpikeTensor(torch.zeros((2,36,36,self.nTimeBins)), samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1

        return inputSpikes, classLabel

    def __len__(self):
        return self.samples.shape[0]