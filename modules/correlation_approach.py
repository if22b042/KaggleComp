import pandas as pd

def correlation_approach(test_data, correlation):
    result_array = []
    for i in range(len(test_data)):
        score = 0
        row = test_data.iloc[i]
        for x in range(len(row)):  # Dynamically iterate over the features
            if correlation[x] > 0.2 or correlation[x] < -0.2:  # Threshold on correlation
                value = row.iloc[x]
                corr_value = correlation.iloc[x]
                score += value * corr_value
        
        result = score > 0
        result_array.append([i, result])
    
    return result_array
