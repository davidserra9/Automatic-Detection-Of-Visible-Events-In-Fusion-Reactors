import json

def saveData(path, data):

    with open(str(path), 'w') as metadataFile:
        json.dump(data, metadataFile)


def loadData(path):
    with open(str(path), "r") as metadataFile:
        data = json.load(metadataFile)
    return data