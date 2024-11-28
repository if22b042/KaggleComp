
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv('test_set.csv', index_col=0)
train_data= pd.read_csv('train_set.csv', index_col=0)

correlation_matrix = train_data.corr()
correlation= correlation_matrix['target']

def bar_graph(first_column):
    plt.figure(figsize=(10, 6))
    first_column.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Correlation of Features with {'target'}")
    plt.xlabel("Features")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

correlation = correlation.iloc[1:]
bar_graph(correlation)
i=0
result_array=[]
while (i<4000):
    score=0
    row=test_data.iloc[i]
    x=0
    while(x<30):
        if(correlation[x]>0.2 or correlation[x]<(-0.2)  ):
            value=row[x]
            corr_value=correlation[x]
            score=score+(value*corr_value)
        x=x+1
        
    result=False
    if(score>0):
        result=True
    
    a=[i, result]
    result_array.append(a)

    i=i+1

result_df = pd.DataFrame(result_array, columns=["ID", "target"])
result_df.to_csv("results.csv", index=False)    


