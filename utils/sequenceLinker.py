def seq2num(seq):
    """
    Link a string sequence filename to a number
    :param seq: string with the filename
    :return: int
    """
    num = -1
    if '20171207.039_AEQ11' in seq:
        num = 1
    elif '20171207.043_AEQ11' in seq:
        num = 2
    elif '20180918.036_AEQ50' in seq:
        num = 3
    elif '20180918.038_AEQ50' in seq:
        num = 4
    elif '20180918.040_AEQ50' in seq:
        num = 5
    elif '20180919.007_AEQ40' in seq:
        num = 6
    elif '20180920.034_AEQ11' in seq:
        num = 7
    elif '20181002.028_AEQ20' in seq:
        num = 8
    elif '20181004.038_AEQ10' in seq:
        num = 9
    elif '20181004.038_AEQ20' in seq:
        num = 10
    elif '20181004.038_AEQ40' in seq:
        num = 11
    elif '20181004.046_AEQ20' in seq:
        num = 12
    elif '20181004.046_AEQ40' in seq:
        num = 13
    elif '20181004.046_AEQ50' in seq:
        num = 14

    return num


def num2seq(num):
    """
    Link every number between 1 and 14 to a sequence name
    :param num: int
    :return: string with the linked filename
    """
    seq = ''
    if num == 1:
        seq = '20171207.039_AEQ11'
    elif num == 2:
        seq = '20171207.043_AEQ11'
    elif num == 3:
        seq = '20180918.036_AEQ50'
    elif num == 4:
        seq = '20180918.038_AEQ50'
    elif num == 5:
        seq = '20180918.040_AEQ50'
    elif num == 6:
        seq = '20180919.007_AEQ40'
    elif num == 7:
        seq = '20180920.034_AEQ11'
    elif num == 8:
        seq = '20181002.028_AEQ20'
    elif num == 9:
        seq = '20181004.038_AEQ10'
    elif num == 10:
        seq = '20181004.038_AEQ20'
    elif num == 11:
        seq = '20181004.038_AEQ40'
    elif num == 12:
        seq = '20181004.046_AEQ20'
    elif num == 13:
        seq = '20181004.046_AEQ40'
    elif num == 14:
        seq = '20181004.046_AEQ50'

    return seq