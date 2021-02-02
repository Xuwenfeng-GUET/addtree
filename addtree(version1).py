class Node:
    
    def __init__(self, value, left=None, right=None):
        self.value = value   # 节点的值
        self.left = left     # 左子节点
        self.right = right   # 右子节点
def grabTree(filename):
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()
def likelihoodMediboostChooseFeat(x, y,funcValue,weights,colIdx,verbose = 1):
    #colIdx = 3
    # Initialize variables
    cutPoint = []
    cutCategory = []
    
    # Compute current Score
    firstDer = -2000 * y / (1 + np.exp(2 * y * funcValue))
    #mu = np.mean(firstDer, axis=0)
    #sigma = np.std(firstDer, axis=0)
    #firstDer =  (firstDer - mu) / (sigma + 0.00001)
    y = -firstDer
    y = y.T
    if len(colIdx) == 0:
        return
    x1 = x[:, colIdx]

    from sklearn import tree
    import re
    clf = tree.DecisionTreeClassifier(min_samples_split = 2,max_depth=1,class_weight='balanced')
    y = y.astype(np.int32)
    clf = clf.fit(x1, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    if dot_data.find('X[') == -1:
        fIdx = None
    else:
        arr_sub = dot_data.split('\\')
        sub_dotdata = arr_sub[0]
        arr_sub = sub_dotdata.split(' ')
        
        #ind = dot_data.index('X<= ')
        #idlist = filter(lambda x: x < x2[ind + 3,0], x2);
        #fIdx = max(idlist)
        cutPoint = arr_sub[-1]
        arr_sub = arr_sub[-3].split('[')
        idxstr = arr_sub[-1].replace(']','')
        fIdx = colIdx[int(idxstr)]
    #fIdx = colIdx
    cutCategory = 1
    #cutCategory = 
    node = {}
    node['fIdx']=fIdx
    node['cutPoint'] =cutPoint
    node['cutCategory'] =cutCategory
    return node


def likelihoodMediboostSplitNode(x,y,weights,obsValues,nodeValue,colIdx,depth,learningRate,membership,min_membership,stepspastminmembership,gamma,update,hessian,minhs):
    node = {}
    node['terminal'] = 1
    node['fIdx'] = None
    node['cutPoint'] = None
    node['cutCategory'] = None
    node['weights'] = weights
    node['obsValues'] = obsValues
    node['value'] = nodeValue
    node['left'] = None
    node['right'] = None
    #node['pre'] = pre
    node['membership'] = membership
    node['depth'] = depth
    min_update = 1000
    if depth < 10 and len(colIdx) > 0 and hessian >= minhs:
        node['terminal'] = 0
        nodefeat = likelihoodMediboostChooseFeat(x, y,node['obsValues'],node['weights'],colIdx)
        node['fIdx'] = nodefeat['fIdx']
        node['cutPoint'] = nodefeat['cutPoint']
        node['cutCategory'] = nodefeat['cutCategory']
        #node['tree'] = nodefeat['tree']
        
        node['membership'] = membership
        leftIdx = []
        rightIdx = []
        if node['cutPoint']  and node['cutCategory']:
            node['terminal'] = 0
            if node['cutPoint']:
                leftIdx = np.where(x[:, node['fIdx']] < float(node['cutPoint']))
                rightIdx = np.where(x[:, node['fIdx']] >= float(node['cutPoint']))            
                #if node['cutPoint'] in x[:, node['fIdx']]:
                #    cutpos = list(x[:, node['fIdx']]).index(node['cutPoint'])
                #    leftIdx = list(range(1,cutpos))
                #    rightIdx = list(range(cutpos,x[:, node['fIdx']].shape[0]))
            
            else:
                'aaa'
            #    if node['cutCategory'] in x[:, node['fIdx']]:
            #        cutpos = list(x[:, node['fIdx']]).index(node['cutCategory'])
            #        leftIdx = cutpos
            #        rightIdx = list(range(cutpos,x[:, node['fIdx']].shape[0]))
            membershipRight = []
            membershipLeft = []
            membershipRight = rightIdx[0]
            membershipLeft = leftIdx[0]
            # Assign new memberships
            # Calculate the observations, weights and node coefficients
            # Calculate left weights and outputs
            leftY = y[leftIdx]
            leftWeight = weights[leftIdx]
            # Calculate right weights and outputs
            rightY = y[rightIdx]
            rightWeight = weights[rightIdx]

            # Calculate the current value of the function for right and left node
            # Vector, length < y
            nodeobs = node['obsValues'][0]
            funcValueLeft = nodeobs[leftIdx]
            funcValueRight = nodeobs[rightIdx]

            # Compute the coefficients of the left child node
            # Vector, length < y
            firstDerLeft = -2000 * leftY / (1 + np.exp(2 * leftY * funcValueLeft))
            # Remove NaNs (from when there are no cases in node)
            #firstDerLeft[is.na(firstDerLeft)] = 0
            # Scalar
            weightedFirstDerLeft = np.dot(np.array(leftWeight).T, np.array(firstDerLeft))
            # Vector
            secDerLeft = abs(firstDerLeft) * (2 - abs(firstDerLeft))
            weightedSecDerLeft = np.dot(np.array(leftWeight).T, secDerLeft)
            if (weightedFirstDerLeft == 0):
                nodeValueLeft = 0
            else:
               if (weightedSecDerLeft == 0):
            # nodeValueLeft <- c(sign(y[leftIdx] %*% leftWeight) * Inf)
                   nodeValueLeft = -np.sign(weightedFirstDerLeft) * float("inf")
               else:
            # 12.31.2017 add learningRate
                updateVal = learningRate * np.sign((weightedFirstDerLeft)) * min(min_update, abs(weightedFirstDerLeft / weightedSecDerLeft))
                nodeValueLeft = node['value'] - updateVal
            observValuesLeft = funcValueLeft - nodeValueLeft

            # Compute the coefficients for the right child node
            firstDerRight = -2000 * rightY / (1 + np.exp(2 * rightY * funcValueRight))
            #firstDerRight[ is.na(firstDerRight)] < - 0
            secDerRight = abs(firstDerRight) * (2 - abs(firstDerRight))
            weightedFirstDerRight =  np.dot(np.array(rightWeight ).T,   firstDerRight)
            weightedSecDerRight = np.dot(np.array(rightWeight ).T ,  secDerRight)
            if (weightedFirstDerRight == 0):
                nodeValueRight = 0
            else:
                 if (weightedSecDerRight == 0):
            # nodeValueRight <- c(sign(y[rightIdx] %*% rightWeight) * Inf)
                        nodeValueRight = -np.sign(weightedFirstDerRight) * float("inf")
                 else:
                    updateVal = learningRate * np.sign(weightedFirstDerRight) * min(min_update, abs(weightedFirstDerRight / weightedSecDerRight))
                    nodeValueRight = node['value'] - updateVal
            observValuesRight = funcValueRight - nodeValueRight
            newObservValue = node['obsValues'].T
            newObservValue[leftIdx] = observValuesLeft.reshape(observValuesLeft.shape[0],1)
            newObservValue[rightIdx] = observValuesRight.reshape(observValuesRight.shape[0],1)
            if (update == "polynomial"):
                leftRule = y
                leftRule[leftIdx] = 1
                leftRule[rightIdx] = gamma

                # Assign 1 to the samples in the right branch and gamma to the remaining
                rightRule = y
                rightRule[rightIdx] = 1
                rightRule[leftIdx] = gamma

                # Update the weights
                leftWeights = node['weights'] * leftRule
                rightWeights = node['weights'] * rightRule
            else:
                # Exponential
                leftRule = y
                leftRule[leftIdx] = 1
                leftRule[rightIdx] = -1
                rightRule = y
                rightRule[leftIdx] = -1
                rightRule[rightIdx] = 1

                lamda = gamma/(1 - gamma)
                if 1:
                  tmpLeft = (np.exp((leftRule - 1)*lamda/2) /(np.exp((leftRule - 1)*lamda/2) + np.exp((rightRule - 1)*lamda/2) ))
                  tmpRight = (np.exp((rightRule - 1)*lamda/2) / (np.exp((leftRule - 1)*lamda/2) + np.exp((rightRule - 1)*lamda/2)))
                  leftWeights = np.array(node['weights'])*(tmpLeft.reshape(tmpLeft.shape[0],1))
                  rightWeights = np.array(node['weights'])*(tmpRight.reshape(tmpRight.shape[0],1))
                else:
                  rightWeights = node['weights']
                  leftWeights = rightWeights
            
            leftWeights = leftWeights / sum(leftWeights)
            rightWeights = rightWeights / sum(rightWeights)
            colIdxRight = [i for i in range(int(node['fIdx'])+1,colIdx[-1])]
            colIdxLeft = [i for i in range(colIdx[0],int(node['fIdx']))]
            nodelist = Node(node['fIdx'])
            node['right'] = likelihoodMediboostSplitNode(x, y,rightWeights,newObservValue.T,nodeValueRight,colIdxRight,depth + 1,learningRate,membershipRight,min_membership,stepspastminmembership,gamma,update,weightedSecDerRight,minhs)
            node['left'] = likelihoodMediboostSplitNode(x, y,leftWeights,newObservValue.T,nodeValueLeft,colIdxLeft,depth + 1,learningRate,membershipLeft,min_membership,stepspastminmembership,gamma,update,weightedSecDerLeft,minhs)
        else:
           node['terminal'] = 1
    else:
        return node
    return node
def classify(inputTree, featLabels, testVec):
	"""
	输入：决策树，分类标签，测试数据
	输出：决策结果
	描述：跑决策树
	"""
	featIndex = list(inputTree.keys())[0]
	secondDict = inputTree[featIndex]
    
	classLabel = 0
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def predict(inputTree, featLabels, testDataSet):
	"""
	输入：决策树，分类标签，测试数据集
	输出：决策结果
	描述：跑决策树
	"""
	classLabelAll = []
	for testVec in testDataSet:
		classLabelAll.append(classify(inputTree, featLabels, testVec))
	return classLabelAll
def predict(inputTree,testDataSet):
    fIdx = inputTree['fIdx']
    membercount = inputTree['left']['membership'].shape[0]
    m = testDataSet.shape[0]
    predicted = []
    for i in range(0,m):
        if i <= membercount:
            predicted.append(0)
        else:
            predicted.append(1)
    return predicted
def mediboost(x,y,w,gamma,N):
    y0 = y
    dt = [x,y]
    probability = np.array(w)/sum(w)
    probability = probability.T
    y = np.array(y)
    nodeValue = np.log(1 + np.dot(probability,y)) - np.log(1 - np.dot(probability,y))
    obsValues = np.ones((1,N))*nodeValue
    learningRate = 0.01
    update = 'exponential'
    colIdx = [i for i in range(0,x.shape[1])]
    membership = np.ones((y.shape[0],1))
    minhs = -float('inf')
    minmembership = 0
    stepspastminmembership = 2
    hessian = float('inf')
    tree = likelihoodMediboostSplitNode(x,y,w,obsValues,nodeValue,colIdx,0,learningRate,membership,minmembership,stepspastminmembership,gamma,update,hessian,minhs)
    return tree
def treecl(node):
    flg1,flg2,flg3,flg4 = 0,0,0,0
    if node['left'] == None:
        flg1 = 1
    else:
        if node['left']['fIdx'] == None:
            flg1 = 1
    
    if node['right'] != None:
        if node['right']['fIdx'] != None:
            flg2 = 1
    if flg2 == 1:
        if node['right']['left'] == None:
            flg3 = 1
        else:
            if node['right']['left']['fIdx'] == None:
                flg3 = 1
        if node['right']['right'] != None:
            if node['right']['right']['fIdx'] != None:
                flg4 = 1
    if flg1 == 1 and flg2 == 1 and flg3 == 1 and flg4 == 1:
                node['left'] = node['right']['right']
                node['right']['right'] = None
    
    flg1,flg2,flg3,flg4 = 0,0,0,0
    if node['right'] == None:
        flg1 = 1
    else:
        if node['right']['fIdx'] == None:
            flg1 = 1
    
    if node['left'] != None:
        if node['left']['fIdx'] != None:
            flg2 = 1
    if flg2 == 1:
        if node['left']['right'] == None:
            flg3 = 1
        else:
            if node['left']['right']['fIdx'] == None:
                flg3 = 1
        if node['left']['left'] != None:
            if node['left']['left']['fIdx'] != None:
                flg4 = 1
    if flg1 == 1 and flg2 == 1 and flg3 == 1 and flg4 == 1:
                node['right'] = node['left']['left']
                node['left']['left'] = None
    flg1,flg2,flg3,flg4 = 0,0,0,0
    if node['right'] == None:
        flg1 = 1
    else:
        if node['right']['fIdx'] == None:
            flg1 = 1
    
    if node['left'] != None:
        if node['left']['fIdx'] != None:
            flg2 = 1
    if flg2 == 1:
        if node['left']['left'] == None:
            flg3 = 1
        else:
            if node['left']['left']['fIdx'] == None:
                flg3 = 1
        if node['left']['right'] != None:
            if node['left']['right']['fIdx'] != None:
                flg4 = 1
    if flg1 == 1 and flg2 == 1 and flg3 == 1 and flg4 == 1:
                node['right'] = node['left']['right']
                node['left']['right'] = None                
    flg1,flg2,flg3,flg4 = 0,0,0,0
    if node['left'] == None:
        flg1 = 1
    else:
        if node['left']['fIdx'] == None:
            flg1 = 1
    
    if node['right'] != None:
        if node['right']['fIdx'] != None:
            flg2 = 1
    if flg2 == 1:
        if node['right']['right'] == None:
            flg3 = 1
        else:
            if node['right']['right']['fIdx'] == None:
                flg3 = 1
        if node['right']['left'] != None:
            if node['right']['left']['fIdx'] != None:
                flg4 = 1
    if flg1 == 1 and flg2 == 1 and flg3 == 1 and flg4 == 1:
                node['left'] = node['right']['left']
                node['right']['left'] = None
    return node

def eee(node):
    if node != None:
        node = treecl(node)
    if node['left']:
        node['left'] = eee(node['left'])
    if node['right']:
        node['right'] = eee(node['right'])
    return node
def AddTree(trdt,tedt,w,lamda,T,v,fn,n  = 1):
   #     x0 = traindt.T[np.arange(1,219)]
   # x = x0.T
   # x_test = testdt.T[np.arange(1,219)]
    x = trdt[:,0:-1]
    y = trdt[:,-1]
    x_test = tedt[:,0:-1]
    y_test = tedt[:,-1]
    y_avg = 0.5
    N = x.shape[0]
    gamma = 0.95
    tree = mediboost(x,y,w,gamma,N)
    #predicted = predict(mod, y, x_test)
    #prune
    
    #node = list_node(tree)
    #node2 = eee(tree)
    #node = list_node(node2)
    return tree

def create_graph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.value] = (x, y)


    
    if node != None and node.left != None and node.value != None and node.left.value != None:

        if node.left != None and node.left.value != None:
            G.add_edge(node.value, node.left.value)
            l_x, l_y = x - 1/2 ** layer, y - 1
            l_layer = layer + 1
            create_graph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node != None and node.right != None and node.value != None:
        if node.right != None and node.right.value != None:
            G.add_edge(node.value, node.right.value)
            r_x, r_y = x + 1/2  ** layer, y - 1
            r_layer = layer + 1
            create_graph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return (G, pos)

def draw(node):   # 以某个节点为根画图
    graph = nx.DiGraph()
    #graph = {2481:{948:{107:{},841:{}},1533:{}}}
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(8, 10))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=300)
    plt.show()
def list_node(tree,nodevalper=''):
    flg = 0
    flgleft = 0
    flgright = 0
    
    if tree:
        if tree['depth'] == 0:
            nodevalper = 'root'
        if tree['fIdx'] != None:
            nodevalleft = "x[" + str(tree['fIdx']) + "]" + "<=" + str(tree['cutPoint'])
            nodevalright = "x[" + str(tree['fIdx']) + "]" + ">" + str(tree['cutPoint'])
            flg = 1
        else:
            a = 1
        if tree['left'] == None:
            flgleft = 1
        else:
            if tree['left']['fIdx'] == None:
                flgleft = 0
        if tree['right'] == None:
            flgright = 1
        else:
            if tree['right']['fIdx'] == None:
                flgright = 0
                      
        if nodevalper == '':
            node = Node(tree['fIdx'])
        else:
            node = Node(nodevalper)
        if flgleft == 0 and flg == 1:
            node.left = list_node(tree['left'],nodevalleft)
        else:
            node.left = list_node(tree['left'])
        if flgright == 0 and flg == 1:
            node.right = list_node(tree['right'],nodevalright)
        else:
            node.right = list_node(tree['right'])
        
    #if (node.left.value != None and node.right.value != None) or (node.left.value == None and node.right.value == None):
    else:
        node = None 
    return node
    #else:
    #    node.left.value = None
    #    node.right.value = None
    
        
def freq(mtx):
    count00 = 0
    count01 = 0
    count10 = 0
    count11 = 0
    mtx = mtx.T
    for i in range(0,mtx.shape[0]):
        if mtx[i,0] == 0 and mtx[i,1] == 0:
            count00 = count00 + 1
        if mtx[i,0] == 0 and mtx[i,1] == 1:
            count01 = count01 + 1
        if mtx[i,0] == 1 and mtx[i,1] == 0:
            count10 = count10 + 1
        if mtx[i,0] == 1 and mtx[i,1] == 1:
            count11 = count11 + 1
    tbl = [[count00,count01],[count10,count11]]
    return tbl
        
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import operator
    import math
    traindt=pd.read_csv('F:/数据集/MalDroid-2020/140D/train2/SMSmalware_train.csv')
    testdt=pd.read_csv('F:/数据集/MalDroid-2020/140D/test2/SMSmalware_test.csv')
    traindt = np.array(traindt)
    traindt = traindt[np.argsort(traindt[:,-1])]
    testdt = np.array(testdt)
    testdt = testdt[np.argsort(testdt[:,-1])]
    w = np.mat(np.ones((traindt.shape[0], 1))/traindt.shape[0])
    w = np.ones((traindt.shape[0], 1))
    lamda = 0
    n = 1
    T = 10
    v = 0.1
    fn = 1
    tree = AddTree(traindt,testdt,w,lamda,T,v,fn,n)
    
    node = list_node(tree)
    draw(node)
    x_test = testdt[:,0:-1]
    featlabels = traindt[:,-1].tolist()
    predicted = predict(tree, x_test)
    predicted = np.array(predicted).T
    
    #stat
    print('training feature',traindt.shape[0],'x',traindt.shape[1]-1)
    print('training outcome',traindt.shape[0],'x',1)
    print('testing feature',testdt.shape[0],'x',testdt.shape[1]-1)
    print('testing outcome',testdt.shape[0],'x',1)
    print('addtree classification training summary:')
    tbl = freq(np.array([predicted,testdt[:,-1]]))
    tbl = np.array(tbl)
    Totals = sum(tbl.T)
    Predicted_totals = sum(tbl)
    Total = sum(sum(tbl))
    Hits = np.diagonal(tbl)
    Misses = Totals - Hits
    Sensitivity = Hits/Totals
    ovrlSensitivity = Sensitivity[1]
    Condition_negative = Total - Totals
    True_negative = Total - Predicted_totals - (Totals - Hits)
    Specificity = True_negative / Condition_negative
    ovrlSpecificity  = Specificity[0]
    Balanced_Accuracy = 0.5*(Sensitivity + Specificity)
    ovrlBalanced_Accuracy = Balanced_Accuracy[0]
    PPV = Hits/Predicted_totals
    ovrlPPV = PPV[0]
    NPV = True_negative/(Total - Predicted_totals)
    ovrlNPV = NPV[1]
    F1 = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
    ovrlF1 = F1[0]
    ovrlAccuracy = sum(Hits)/Total
    print('Sensitivity = '+str(ovrlSensitivity))
    print('Specificity = '+str(ovrlSpecificity))
    print('Balanced_Accuracy = '+str(ovrlBalanced_Accuracy))
    print('PPV = '+str(ovrlPPV))
    print('NPV = '+str(ovrlNPV))
    print('F1 = '+str(ovrlF1))
    print('Accuracy = '+str(ovrlAccuracy))
    #print('Sensitivity Specificity Balanced_Accuracy PPV NPV F1 Accuracy')
    #print(ovrlSensitivity,ovrlSpecificity,ovrlBalanced_Accuracy,ovrlPPV,ovrlNPV,ovrlF1,ovrlAccuracy)
    #sensititvy https://zhuanlan.zhihu.com/p/137953265
    #https://blog.csdn.net/sunflower_sara/article/details/81214897
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("accuracy is %.4f"%accuracy_score(testdt[:,-1], predicted))
    print("precision is %.4f"%precision_score(testdt[:,-1], predicted))
    print("recall is %.4f"%recall_score(testdt[:,-1], predicted))
    print("f1_score is %.4f"%f1_score(testdt[:,-1], predicted))