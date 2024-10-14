import numpy as np
import pandas as pd
import cv2
import os
import glob
from sklearn.model_selection import KFold
import datetime
from sklearn.metrics import (mean_squared_error,mean_absolute_error,mean_absolute_percentage_error)



def print_results(lst_mse,lst_rmse,lst_mae,lst_mape,lst_times,model_name):    

    results_path=f'./results_{model_name}.txt'
    f1=open(results_path,'a')

    f1.write('\nAverage MSE: '+str(np.mean(lst_mse)))
    f1.write('\nAverage RMSE: '+str(np.mean(lst_rmse)))
    f1.write('\nAverage MAE: '+str(np.mean(lst_mae)))
    f1.write('\nAverage MAPE: '+str(np.mean(lst_mape)))
    f1.write('\nAverage Training Time: '+str(np.mean(lst_times)))

    f1.write('\n\n\nMetrics for all Folds: \n')
    for i in range(len(lst_mse)):
        f1.write('\n MSE: '+str(lst_mse[i]))
        f1.write('\n RMSE: '+str(lst_rmse[i]))
        f1.write('\n MAE: '+str(lst_mae[i]))
        f1.write('\n MAPE: '+str(lst_mape[i]))
        f1.write('\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\n___________________\n')
    f1.close()



def outlier_detection(x_data,y_data):

    from sklearn.ensemble import IsolationForest as outlier_detector
    model=outlier_detector()
    predicts=model.fit_predict(x_data,y_data)
 
    x_data2=[]
    y_data2=[]
    for i in range(len(predicts)):
        if predicts[i]==1:
            x_data2.append(x_data[i])
            y_data2.append(y_data[i])

    x_data=np.array(x_data2)
    y_data=np.array(y_data2)
    return x_data,y_data


def create_model():
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras.applications import ResNet101V2 as pre_trained_net

    base_model = pre_trained_net(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    model_output = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=model_output)

    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model_name = 'ResNet101V2'
    return model, model_name



def read_data():

    # Load images and labels
    main_path='./Converted Images/'
    image_addresses=glob.glob(main_path+'*.png')

    images=[]
    labels=[]
    for address in image_addresses:
        
        # Load labels (age)
        image_name=os.path.splitext(os.path.basename(address))[0]
        age=image_name.split('Age')[1].split('_')[0]
        if age!='None':
            labels.append(age)
        
            # Load images
            img=cv2.imread(address)
            img = cv2.resize(img, (224, 224))
            img=img.astype('float64')
            img=(img-np.min(img))/(np.max(img)-np.min(img))
            images.append(img)

    return np.array(images),np.array(labels)


# Laod data and preprocess
images,labels=read_data()
labels=labels.astype('float')
print('Data shape: ',images.shape,labels.shape)

# Outlier detection
image_shape=images.shape
images=np.reshape(images, (images.shape[0],images.shape[1]*images.shape[2]*images.shape[3]))
images,labels=outlier_detection(images,labels)
images=np.reshape(images, (images.shape[0],image_shape[1],image_shape[2],image_shape[3]))
print('Data shape after outlier detection: ',images.shape,labels.shape)


lst_mse,lst_rmse,lst_mae,lst_mape,lst_times=[],[],[],[],[]
fold_number=1
kf=KFold(n_splits=5,random_state=42,shuffle=True)
for train ,test in kf.split(images, labels): 

    x_train=images[train]
    x_test=images[test]
    y_train=labels[train]
    y_test=labels[test]

    # Load model
    model,model_name=create_model()

    # Prepare the model
    model.fit(x_train,y_train, epochs=10, batch_size=128)
    for layer in model.layers[:-10]:
        layer.trainable = False
    for layer in model.layers[-10:]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])

    # Train the model and make predictions
    start_time=datetime.datetime.now()
    model.fit(x_train,y_train, epochs=20 ,batch_size=128)
    end_time=datetime.datetime.now()
    training_time=end_time-start_time

    predicts=model.predict(x_test)
    actuals=y_test
    model.save(f'./results/{model_name}_fold{fold_number}.h5')

    # Calculate evaluation metrics
    lst_mse.append(mean_squared_error(actuals,predicts))
    lst_rmse.append(mean_squared_error(actuals,predicts)**0.5)
    lst_mae.append(mean_absolute_error(actuals,predicts))
    lst_mape.append(mean_absolute_percentage_error(actuals,predicts))
    lst_times.append(training_time)

    # Save predictions (actuals and predicts)
    predicts_path=f'./results/{model_name}_predicts_fold{fold_number}.xlsx'
    actuals=pd.DataFrame(actuals)
    predicts=pd.DataFrame(predicts)
    merged_df=pd.concat([actuals,predicts],axis=1)
    merged_df.to_excel(predicts_path)

    fold_number+=1


# Save results
print_results(lst_mse,lst_rmse,lst_mae,lst_mape,lst_times,model_name)
