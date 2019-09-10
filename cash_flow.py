import pandas as pd
import numpy as np


def cal_csv(in_csv='e:\st_score.csv',out_csv='e:\st_model.csv'):
    score = pd.read_csv(in_csv)

    # print(score.head(3))

    print('shape',score.shape)
    for i in range(score.shape[0]):
        l = score.iloc[i,1:7]
        u = l.unique()

        len_u = len(u)
        score.iloc[i,7] = len_u

        model_count = {}
        for j in range(len_u):
            key = '%d'%u[j]
            model_count[key] = 0
            

        label_index=0
        
        label_count_list = np.zeros((6,2))
        label = score.iloc[i,1]  # a score(eg 5) is a label
        label_count_list[label_index] = [label,0]
        
        for j in range(7):
            if j==0: #skip the first item,the first item is name of stocks (eg.*ST神城)
                continue

            if label == score.iloc[i,j]:
                label_count_list[label_index] += [0,1]
            else:
                label = score.iloc[i,j]
                label_index +=1
                label_count_list[label_index] = [label,0]
                label_count_list[label_index] += [0,1]
            model_count['%d'%label] += 1   

        #max continue sequence
        max_v = np.max(label_count_list[:,1])

        score.iloc[i,8] = max_v


        #max occurence
        #print('model_count',model_count)

        max_occ = max(model_count.values())

        occ_cvs_index = 9
        for key,value in model_count.items():
            if value== max_occ:
                score.iloc[i,occ_cvs_index] = int(key)
                occ_cvs_index = 10

        print('\r',i,end='')
    

    print(score.head(10))

    score.to_csv(out_csv,encoding="gbk")
    print('save csv file to', out_csv)

cal_csv()
cal_csv(in_csv='e:\good_score.csv', out_csv='e:\good_model.csv')