# Conversion-fraud-in-Digital-Advertising

### About the project
In Digital advertisement, the goal is to change the visiting people into leads. But some times, some agencies use fake leads which is not good. 
So with the helpf this data, I tried to find out about that fraud. 


### Tool Used
Used Python for data analysis for that particular dataset. Libraries used for this project are:
Numpy
Pandas
Matplotlib
Seaborn
sklearn

### Installation
for numpy
```pip install numpy```


for Pandas
```pip install pandas```


For Matplotlib
```pip install matplotlib```


For Seaborn
```pip install seaborn```

For Sklearn
```pip install scikit-learn```



### Key Insights

I cleaned the data as it had some Nan values and special characters.
Also used feature engineering as there were few features which were not that useful for model.
Used few different algoritms for prediction and on the basis of overall performance choosed the final model.
The one example of KNN. The code i wrote for KNN was:

```
# As we are uncertain about number of neighbors so will use it with as list and compare the errors
error_rate = []
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
        
#Plotting the errors
plt.plot(range(1,15),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```


And the output of above code was:

<img width="600" alt="k" src="https://user-images.githubusercontent.com/69238621/140892870-b4760de5-f08c-4fb8-a3a0-bf8e7f8f6c1d.PNG">

We select the value of K from the plot. It is that value which is lowest and after that it does nt matter if nex value rise or fall but not lower than the chosen K.
