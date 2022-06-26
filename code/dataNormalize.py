from sklearn.preprocessing import StandardScaler

def dataNormalize(ndarray):
    scaler = StandardScaler() 
    data_scaled = scaler.fit_transform(ndarray)
    return data_scaled
