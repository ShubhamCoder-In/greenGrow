import pandas as pd 
import joblib 

lb1 = joblib.load('./model/issue_tla_lb.pkl')
lb2 = joblib.load('./model/tla_lb.pkl')
model_pipeline = joblib.load('./model/model_pipeline.pkl')

val_df = pd.read_csv('./data/test_data.csv')
filter_data = pd.read_csv('./data/filter_data.csv')

def prediction_ouput(input):
    ped_df = pd.DataFrame([])
    prediction = 0.0
    status = 404
    if input['year'] >2022 and input['year']<2100:
        predict_year = input['year']+1
        Epi_value_1= val_df[
            (val_df['code']==input['code'])&(val_df['TLA']==input['TLA'])&(val_df['issue_tla']==input['issue_tla'])&(val_df['year']==2022)
            ]['Epi_value'].values[0]
        
        Epi_value_2 = val_df[
            (val_df['code']==input['code'])&(val_df['TLA']==input['TLA'])&(val_df['issue_tla']==input['issue_tla'])&(val_df['year']==2021)
            ]['Epi_value'].values[0]

        input['TLA'] = lb2.transform([input['TLA']])
        input['issue_tla'] = lb1.transform([input['issue_tla']])
        for year in range(2023,predict_year):
            df = pd.DataFrame([],columns=['code', 'TLA', 'issue_tla', 'year', 'Epi_value_1', 'Epi_value_2'])
            df.loc[0,'code'] = input['code']
            df.loc[0,'TLA']= input['TLA']
            df.loc[0,'issue_tla'] = input['issue_tla']
            df.loc[0,'year']= year
            df.loc[0,'Epi_value_1'] = Epi_value_1
            df.loc[0,'Epi_value_2'] = Epi_value_2
            data = df[['code', 'TLA', 'issue_tla', 'year', 'Epi_value_1', 'Epi_value_2']]
            prediction = model_pipeline.predict(data)
            df.loc[0,'Epi_value'] = prediction
            ped_df = pd.concat([ped_df,df],axis=0, ignore_index=True)
            Epi_value_2 = Epi_value_1
            Epi_value_1 = prediction
        status = 200
    else:
        if input['year'] >1996 and input['year']<2023:
            Epi_value_1= filter_data[
                (filter_data['code']==input['code'])&(filter_data['TLA']==input['TLA'])&(filter_data['issue_tla']==input['issue_tla'])&(filter_data['year']==input['year'])
                ]['Epi_value_1'].values[0]
            Epi_value_2 = filter_data[
                (filter_data['code']==input['code'])&(filter_data['TLA']==input['TLA'])&(filter_data['issue_tla']==input['issue_tla'])&(filter_data['year']==input['year'])
                ]['Epi_value_2'].values[0]

            input['TLA'] = lb2.transform([input['TLA']])
            input['issue_tla'] = lb1.transform([input['issue_tla']])

            df = pd.DataFrame([],columns=['code', 'TLA', 'issue_tla', 'year', 'Epi_value_1', 'Epi_value_2'])
            df.loc[0,'code'] = input['code']
            df.loc[0,'TLA']= input['TLA']
            df.loc[0,'issue_tla'] = input['issue_tla']
            df.loc[0,'year']= input['year']
            df.loc[0,'Epi_value_1'] = Epi_value_1
            df.loc[0,'Epi_value_2'] = Epi_value_2
            prediction = model_pipeline.predict(df)
            df.loc[0,'Epi_value'] = prediction
            ped_df = pd.concat([ped_df,df],axis=0, ignore_index=True)
            status = 200
        else:  
            df = pd.DataFrame([],columns=['code', 'TLA', 'issue_tla', 'year', 'Epi_value_1', 'Epi_value_2'])
            input['TLA'] = lb2.transform([input['TLA']])
            input['issue_tla'] = lb1.transform([input['issue_tla']])
            df = pd.DataFrame([],columns=['code', 'TLA', 'issue_tla', 'year', 'Epi_value_1', 'Epi_value_2'])
            df.loc[0,'code'] = input['code']
            df.loc[0,'TLA']= input['TLA']
            df.loc[0,'issue_tla'] = input['issue_tla']
            df.loc[0,'year']= input['year']
            df.loc[0,'Epi_value_1'] = 0.0
            df.loc[0,'Epi_value_2'] = 0.0  
            prediction = [[None]]
            df.loc[0,'Epi_value'] = prediction
            ped_df = pd.concat([ped_df,df],axis=0, ignore_index=True)

    return {'prediction': prediction, 'data': ped_df, 'status': status}