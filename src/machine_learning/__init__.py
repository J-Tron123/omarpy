import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

class preprocessing:
    
    def scaler(self, scaler: str, data: np.array):
        """
        Scales the data.

        Args:
            scaler: A scaler to choose from sklearn library. StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
            data: Array like data to be scaled.
        
        Returns:
            The data scaled.
        """
        try:
            if scaler=='StandardScaler':
                return StandardScaler().fit_transform(data)
            elif scaler=='MinMaxScaler':
                return MinMaxScaler().fit_transform(data)
            elif scaler=='MaxAbsScaler':
                return MaxAbsScaler().fit_transform(data)
            elif scaler=='RobustScaler':
                return RobustScaler().fit_transform(data)
            else:
        
        except ValueError:
            print('Choose one of the scalers listed.')