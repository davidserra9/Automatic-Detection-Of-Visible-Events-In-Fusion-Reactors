def getTimestamps (file):
    
    PROJECT_ROOT = '/content/drive/My Drive/TFG/'
    FILENAME = PROJECT_ROOT + 'utils/all_timestamps.txt'

    f = open(FILENAME, "r")

    for line in f:
        if line.__contains__(file):
            aux = line.split(' ')
            T1 = aux[1]
            T4 = aux[2]
            T4e = aux[3]
            break

    f.close()

    return [int(T1),int(T4),int(T4e)]