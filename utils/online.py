from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import math

def normalizeData(X, scaler='Standard', range=(0,1)):
    """
    Function to scale features with different parameters
    :param X: feature matrix
    :param scaler: type of scaler ('Standard', 'MaxAbs', 'MinMax')
    :return X_norm: feature matrix normalized
    """

    X_norm = []

    if scaler=='Standard':
        print('Scaling each feature by removing the mean and scaling to unit variance')
        scaler = StandardScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)

    if scaler=='MaxAbs':
        print('Scaling each feature by its maximum absoulute value.')
        scaler = MaxAbsScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)

    if scaler=='MinMax':
        print('Normalizing the input data such that the min and max value are', range)
        scaler = MinMaxScaler(feature_range=range)
        scaler.fit(X)
        X_norm = scaler.transform(X)
        
    return X_norm
    
    
def onlineClassification(network, classifier, images, n):
    predictions = []
    for numFrames in range(n, len(images)):
        delta = numFrames / (n-1)
        fr = []
        for i in range(n):
            fr.append(math.floor(delta*i))
            
        features = model(images[0]).tolist() + model(images[20]).tolist() + model(images[40]).tolist() + model(images[60]).tolist() + model(images[-1]).tolist()
        flatFeatures = [f for subFeatures in features for f in subFeatures]
        flatFeatures = np.array([flatFeatures])
        features = normalizeData(flatFeatures, scaler='Standard', range=(0,1))
        pred = XGBoost.predict(features)
        print('Frame: {} - {}'.format(numFrame, int(pred[0])))
    
        predictions.append(int(pred[0]))
    return predictions
