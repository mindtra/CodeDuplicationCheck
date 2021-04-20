import os
import sys
import numpy as np
import re


def binFind(numList, num):  # 返回第一个比num大的下标
    left = 0
    right = len(numList) - 1
    while left <= right:
        mid = (left + right) // 2
        if num < numList[mid]:
            right = mid - 1
        elif num > numList[mid]:
            left = mid + 1
        else:
            return mid
    return left


def getLISLength(numList):
    dp = [numList[0]]
    for i in range(1, len(numList)):
        if dp[-1] < numList[i]:
            dp.append(numList[i])
        else:
            pos = binFind(dp, numList[i])
            dp[pos] = numList[i]
    return len(dp)


def LCS2LISLength(s1, s2):  # s1,s2两个字符串
    LISNumList = []
    for i in range(len(s1)):
        indices = [j for j, letter in enumerate(s2) if letter == s1[i]]
        indices.reverse()
        LISNumList.extend(indices)
    if len(LISNumList) != 0:
        return getLISLength(LISNumList)
    return 0


def getLCSLength(s1, s2):
    c = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    maxLength = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                maxLength = max(c[i][j], maxLength)
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return maxLength


def getLevenshteinDistance(s1, s2):
    c = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        c[i][0] = i
    for j in range(len(s2) + 1):
        c[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                c[i][j] = c[i - 1][j - 1]
            else:
                c[i][j] = 1 + min(c[i - 1][j - 1], c[i][j - 1], c[i - 1][j])

    return c[-1][-1]


def getS(code1, code2, funType=1):
    S = [[0] * len(code2) for _ in range(len(code1))]
    for i in range(len(code1)):
        for j in range(len(code2)):
            if funType == 1:
                # LCS
                S[i][j] = getLCSLength(code1[i], code2[j]) / min(len(code1[i]), len(code2[j]))
            elif funType == 2:
                # LCS转为LIS问题求解
                S[i][j] = LCS2LISLength(code1[i], code2[j]) / min(len(code1[i]), len(code2[j]))
            elif funType == 3:
                # LevenshteinDistance 编辑距离
                S[i][j] = 1 - getLevenshteinDistance(code1[i], code2[j]) / max(len(code1[i]), len(code2[j]))
            # 事实上，这里的编辑距离就是最长子串长度减去最长子序列长度
            # 而最长子序列长度可以通过求解LIS问题获得
            elif funType == 4:
                # 所以此处3跟4是等价的。如果在这里有对LCS优化，那么它们就可以应用到此处
                S[i][j] = LCS2LISLength(code1[i], code2[j]) / max(len(code1[i]), len(code2[j]))
    return S


def getD(code1, code2, r=0.8, funType=1):
    """

    :param code1: 代码1
    :param code2: 代码2
    :param r: 查重率
    :return: 矩阵D
    """
    if funType == 1:
        print("普通LCS")
    elif funType == 2:
        print("时间优化的LIS求解LCS")
    elif funType == 3:
        print("编辑距离")
    elif funType == 4:
        print("编辑距离转化为LCS再使用LIS")

    D = [[0] * len(code2) for _ in range(len(code1))]
    S = getS(code1, code2, funType)
    for i in range(len(code1)):
        for j in range(len(code2)):
            if S[i][j] >= r:
                D[i][j] = 1
            else:
                D[i][j] = 0
    return D


def calMaxDuplicate(D):
    dp = [[0] * (len(D[0]) + 1) for _ in range(len(D) + 1)]
    maxNum = 0
    duplicateList = []
    for i in range(1, len(D) + 1):
        for j in range(1, len(D[0]) + 1):
            if D[i - 1][j - 1] == 0:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            else:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                maxNum = max(maxNum, dp[i][j])
    i = len(dp) - 2
    j = len(dp[0]) - 2
    while i >= 0 and j >= 0:
        if D[i][j] == 1:
            duplicateList.append(str(i + 1) + '----' + str(j + 1))
            i -= 1
            j -= 1
        else:
            if dp[i + 1][j + 1] == dp[i][j]:
                i -= 1
                j -= 1
            elif dp[i][j + 1] >= dp[i + 1][j]:
                i -= 1
            else:
                j -= 1

    return maxNum, duplicateList[::-1]


def codeProcess(codes):  # 输入代码
    newCode = []
    nameList = []
    pattern = r'int (.+?)[ \(\)\|&\^,;=\/><\+\-]'  # 找出int 与中括号里这些符号中间的词
    # searchPattern = r'(?<=int )(.+?)(?=[ \|&\^,;=/><\+\-])'
    subName = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1']
    for i in range(len(codes)):
        codes[i] = codes[i].replace('}', '')  # 去掉花括号
        codes[i] = codes[i].replace('{', '')
        codes[i] = codes[i].strip(' \n\t')  # 去掉前后的空格，换行符和tab
        # name = re.search(searchPattern, codes[i]).group()
        name = re.findall(pattern, codes[i])
        nameList.extend(name)  # 把名称存起来，在下文遇到时重复的方便替换

        for j in range(len(nameList)):
            codes[i] = re.sub(r'(?<![\w])(' + nameList[j] + r')(?![\d\w])', subName[j], codes[i])
        if len(codes[i]) != 0 and not codes[i].isspace():  # 有时全部是空格的话，不会被删掉，手动删除
            newCode.append(codes[i])
    return newCode


# htmlStr = "<html><p>welcome to westos!</p></html>"
# pattern = r'<(\w+)><(\w+)>(.+)</\2></\1>'
# print(re.findall(pattern, htmlStr))
# print(re.findall(pattern, htmlStr)[0][2])
# help(re.compile)
"""
1、. 匹配任意除换行符“\n”外zd的字符；
2、*表示匹配前一个字符0次或无限次；
3、+或*后跟？表示非贪婪匹配，即尽可能少的匹配，如版*？重复任意次，但尽可能少重复；
4、 .*? 表示匹配任意数量的重复，但是在能使整个匹配成功的前提下使用最少的重复。
如：a.*?b匹配最短的，以a开始，以b结束的字符串。如果把它应用于aabab的话，它权会匹配aab和ab。
"""
#
if __name__ == '__main__':
    file1 = open(r"A.txt", 'r')
    # file2 = open(r"B.txt", 'r')
    # file2 = open(r"C.txt", 'r')
    file2 = open(r"D.txt", 'r')
    code1 = file1.readlines()
    code2 = file2.readlines()
    print("处理前")
    for code in code1:
        print(code)
    for code in code2:
        print(code)
    print("加入了数据预处理")
    code1 = codeProcess(code1)
    code2 = codeProcess(code2)
    print("funAAA:")
    for code in code1:
        print(code)
    print("funDDD:")
    for code in code2:
        print(code)
    res, list = calMaxDuplicate(getD(code1, code2, r=0.8, funType=1))
    print("重复代码数：")
    print(res)
    print("重复代码对应的行")
    print(list)
    # numList = [2, 1, 5, 3, 6, 4, 6, 3]
    # print(getLISLength(numList))
