
# call by numpy, time 中 sleep 라이브러리 

from numpy import *
from time import sleep
import matplotlib.pyplot as plt

# 데이터 로드 함수 생성
# 파일을 열고 각각의 줄을 분류 항목 표시와 데이터 행렬로 구문 분석

def loadDataSet(fileName):
    datamat = [] 
    labelmat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        datamat.append([float(lineArr[0]), float(lineArr[1])])
        labelmat.append(float(lineArr[2]))
    return datamat,labelmat


# 첫번째 변수 i = 첫번째 알파의 색인, m = 알파 전체의 개수
# 임의로 선택된 하나의 값은 i와 동일하지 않을 때까지 반복하여 선택한 값으로 반환함
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


# 알파 값이 H보다 크거나 t보다 작은 값일 경우 더 크거나 더 작아지지 않도록 고정시킴
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# datamatin = 데이터 집합, classlabels = 분류 항목 표시, C = 상수 C, toler = 오차범위(tolerance), maxiter = 종료 전 최대 반복 횟수
# 
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)      # datamatin의 형태로부터 상수 m, n을 구함
    alphas = mat(zeros((m,1)))   # 알파를 위한 Column matrix 생성, 0으로 초기화
    iter = 0                     # 어떤한 알파의 변경도 없이, 데이터 집합을 처리하는 반복 횟수를 세는 변수
    while (iter < maxIter):
        alphaPairsChanged = 0    # 각각의 반복은 alphapairschanged를 0으로 설정한 다음 순차적으로 전체 집합을 통과 시킴
        # alphapairschanged는 어떤 알파의 최적화를 시도할 경우 이를 기록하는데 사용됨
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 분류 항목에 대한 예측인 fxi를 계산
            Ei = fXi - float(labelMat[i])  # 오류 Ei는 예측과 실제 분류 항목을 기반으로 계산
            # 알파가 0 또는 C와 동일하다면, 이 알파는 bound가 되므로 0 또는 C에 고정됨
            # 이 알파는 더 이상 증가나 감소가 일어날 수 없으므로 최적화를 시도할 필요가 없어지기 때문
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   
    elif kTup[0]=='rbf': #elif로 커널 유형 추가 가능
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) # "/"표시는 역행렬을 계산하는 것이 아니라 원소 연산을 의미한다는 것
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


# 중요한 값들을 모두 저장해 두기 위한 데이터 구조를 생성한 것
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # eCache 소속 변수 추가, 추가된 소속 변수는 (m by 2) 행렬
                                             # 첫번째 column은 eCache가 유효한지 상태를 나타내는 flag bit, 두번째 Column은 실질적인 E 값
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 주어진 알파에 대한 E값을 계산하고, 계산된 E 값을 반환
            
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        

# 두 번째 알파 또는 내부 반복문의 알파를 선택 ( 두번째 알파를 선택하기 위한 것)

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]  # 첫번째로 선택한 알파인 Ei와 인덱스 i에 관련된 오류 값을 찾음
    # cache에서 유효한 방향으로 입력 Ei를 설정함 ( 유효하다는 것은 계산될 수 있다는 것을 의미)
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] # eCache에 있는 값 중 0이 아닌 값으로 구성된 리스트를 생성
    # 참고로 nonzero()는 numpy함수로 입력 리스트에서 0이 아닌 인덱스만으로 구성된 리스트를 반환
    # nonzero()가 반환하는 값의 형태는 E 값이 아니라, 0이 아닌 E 값이 해당하는 알파임
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   
            if k == i: continue 
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 변환된 모든 알파 값을 반복적으로 처리하여 변화가 가장 큰 값을 선택 (반복문을 처음 실행할 때는 임의로 알파를 선택
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:   
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 오류를 계산하고 계산한 값을 캐시에 저장, 알파 값을 최적화한 후 이 값을 사용

def updateEk(oS, k):  
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        
        # slectJrand() 대신 selectJ()가 두번째 알파를 선택하는 데 사용됨
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: # (임시적으로 없앰) print("L==H"); 
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: #(임시적으로 없앰) print("eta>=0"); 
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) # eCache 갱신
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            #(임시적으로 없앰) print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) # eCache 갱신 
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0


#전체 플랫 SMO 알고리즘    

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = 1
    alphaPairsChanged = 0
    # 반복문은 반복 횟수가 지정한 최대치를 초과할 경우나, 전체 집합에 있는 어떠한 알파 쌍에도 변화가 없을 경우 종료
    # maxIter는 기존 함수에서는 알파가 변하지 않을 때 전체 집합을 가지고 진행한 반복 횟수를 계산하여 저장하는 데 이 변수를 사용,
    # 하지만 이 함수에서는 반복문이 한 번 수행되는 것을 무조건 반복으로 정의 (smosimple()에서 횟수를 세기 위해 사용한 방법보다 우수함)
    # 왜냐하면 최적화에서 이 방법은 어떠한 진동도 없을 때 멈추기 때문
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   
            # 첫 번째 for 반복문은 데이터 집합에 있는 알파를 검토하지 않음, 두 번째 알파를 선택하기 위해 innerL()을 호출하고, 가능하다면
            # 최적화를 수행. 이 함수는 어떤 쌍이든 변화가 발생한다면 1을 반환한다.
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                # (임시적 으로 없앰 )print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        # 두 번째 for 반복문은 범위에 속하지 않는 0 또는 C범위 값을 가지지 않는 모든 알파를 검토
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                #(임시적으로 없앰) print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        #(임시적으로 없앰) print("iteration number: %d" % iter)
    # 마지막으로 상수 b와 알파를 반환함
    return oS.b,oS.alphas

# 알파를 구하기 위해 많은 시간을 보냄, 그런데 분류에 이것들을 어떻게 사용하지?
# 먼저 알파로부터 초평면을 구한 후, 이것은 ws 계산을 수반함

def calcWs(alphas,dataArr,classLabels):
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataArr)
    w = zeros((n,1))
    # 가장 중요한 부분임. 대부분의 알파가 0임, 0이 아닌 알파는 지지벡터임.
    # 따라서 단지 support vector만을 사용함, 따라서 다른 데이터 점들을 쉽게 버릴 수있음, 이것들은 w를 계산하는데 기여하지 않기 때문
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],dataArr[i,:].T)
    return w


# 커널을 이용한 SVMs

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def testRbf(loaddata,testdata,k1=1.3):
    dataArr,labelArr = loadDataSet(loaddata)
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    # 처음 두줄이 포인트임, 커널을 이용해 분류하는 방법을 보여줌,Kerneltrans()를 사용해 변환된 데이터를 구하고 나면, 알파와 분류 항목을 표시한 값을 곱함
    # 주목해야 할 것은 지지 벡터를 위한 데이터 사용 방법임 (나머지 데이터는 버릴 수 있음)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet(testdata)
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    # 검사용 데이터 집합인 다른 데이터 집합으로 반복한다, 검사를 위한 설정과 훈련을 위한 설정이 어떻게 다른지 비교할 수 있음
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))   
        
    
def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas


def testRBF(traindata,trainlabel,testdata,testlabel,k1=1.3):
    b,alphas = smoP(traindata, trainlabel, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(traindata); labelMat = mat(trainlabel).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("Support Vectors          : ",shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    # 처음 두줄이 포인트임, 커널을 이용해 분류하는 방법을 보여줌,Kerneltrans()를 사용해 변환된 데이터를 구하고 나면, 알파와 분류 항목을 표시한 값을 곱함
    # 주목해야 할 것은 지지 벡터를 위한 데이터 사용 방법임 (나머지 데이터는 버릴 수 있음)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(trainlabel[i]): errorCount += 1
    print("training error rate      : %f" % (float(errorCount)/m))
    trainerror=(float(errorCount)/m)
    errorCount = 0
    datMat=mat(testdata); labelMat = mat(testlabel).transpose()
    m,n = shape(datMat)
    # 검사용 데이터 집합인 다른 데이터 집합으로 반복한다, 검사를 위한 설정과 훈련을 위한 설정이 어떻게 다른지 비교할 수 있음
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(testlabel[i]): errorCount += 1    
    print("test error rate          : %f" % (float(errorCount)/m))
    testerror=(float(errorCount)/m)
    return trainerror, testerror
    
    
    loaddata = input('입력 할 데이터 명 : ')
dataarr, labelarr = loadDataSet(loaddata)

def k_split(a,b,hold):
    zete=[]
    bete=[]
    for i in range(hold):
        zete.append(a[i*int(len(a)/hold):i*int(len(a)/hold)+int(len(a)/hold)])
        bete.append(b[i*int(len(a)/hold):i*int(len(a)/hold)+int(len(a)/hold)])
    return zete,bete

holds=int(input(' K (k-fold) : '))
dataarr_split,labelarr_split=k_split(dataarr,labelarr,holds)

def kfolds(zete,j):
    test=[]
    train=[]
    for i in range(len(zete)):
        if i==j:
            test+=zete[i]
        elif i!=j:
            train+=zete[i]
    return test, train

datatest_split=[]
labeltest_split=[]
datatrain_split=[]
labeltrain_split=[]
for i in range(holds):
    datatestset,datatrainset=kfolds(dataarr_split,i)
    datatest_split.append(datatestset)
    datatrain_split.append(datatrainset)
    labeltestset,labeltrainset=kfolds(labelarr_split,i)
    labeltest_split.append(labeltestset)
    labeltrain_split.append(labeltrainset)

count_trainerror=0
count_testerror=0
for i in range(holds):
    print('-----------[',i,']----------------------')
    trainerror, testerror=testRBF(datatrain_split[i],labeltrain_split[i],datatest_split[i],labeltest_split[i],10)
    count_trainerror+=trainerror
    count_testerror+=testerror
print('-------------------------------------')
print('---------[S U M M A R Y ]------------')
print('Training error rate Mean : %f' % (float(count_trainerror/holds)))
print('Test error rate Mean     : %f' % (float(count_testerror/holds)))
