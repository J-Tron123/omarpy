def reg_accuracy(y_true, y_pre):
    return_var = []
    from math import sqrt
    rmse = sqrt(mean_squared_error(y_true, y_pre))
    return_var.append(rmse)
    print ("RMSE: ", rmse)
    r2 = r2_score(y_true, y_pre)
    return_var.append(r2)
    print("R2: ", r2)
    mae = mean_absolute_error(y_true, y_pre)
    return_var.append(mae)
    print("MAE: " , mae)

    if 0 in y_true:
        print("MAPE erróneo")
        return_var.append(0)

    else:
        mape = round(np.mean(np.abs((y_true - y_pre) / y_true))*100,4)
        print('MAPE :', mape)
        print('======================')
        print('Model Accuracy(%) :', 100 - mape)
        print('======================')
        return_var.append(mape)
        return_var.append(100-mape)
    return return_var