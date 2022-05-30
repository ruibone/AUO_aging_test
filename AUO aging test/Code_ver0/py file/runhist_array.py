def runhist_array_f(runhist_array, id_mapping):
    
    new_array = runhist_array.merge(id_mapping, on = 'sheet_id', how = 'left') # all sheet_id in runhist_array are kept
    col = new_array.columns.tolist()
    col = col[5:6] + col[1:4] # discard lot_no
    new_array = new_array[col]
    new_array['new_eqp'] = [x.split('_')[0] for x in new_array.eqp_id_info]
    new_array = new_array.drop(columns = ['create_time', 'eqp_id_info'])

    array_dummy = pd.get_dummies(new_array, columns = ['new_eqp', 'op_id_info']) # eqp_id_info(348/57), op_id_info(54)
    array_dummy = array_dummy.groupby(['id']).max().reset_index()
    
    return array_dummy 


#####Data Processing#####
runhist_array_m1 = pd.read_csv('ARRAY_RunHist/runhist_array_m1.csv')
runhist_array_m2 = pd.read_csv('ARRAY_RunHist/runhist_array_m2m3.csv')
runhist_array_m8 = pd.read_csv('ARRAY_RunHist/runhist_array_m8m9m10m11.csv')
runhist_array_m12 = pd.read_csv('ARRAY_RunHist/runhist_array_m12.csv')
id_mapping_m1 = pd.read_csv('ARRAY_RunHist/id_mapping_m1.csv')
id_mapping_m2 = pd.read_csv('ARRAY_RunHist/id_mapping_m2m3.csv')
id_mapping_m8 = pd.read_csv('ARRAY_RunHist/id_mapping_m8m9m10m11.csv')
id_mapping_m12 = pd.read_csv('ARRAY_RunHist/id_mapping_m12.csv')

runhist_array = pd.concat([runhist_array_m8, runhist_array_m12, runhist_array_m1, runhist_array_m2])
id_mapping = pd.concat([id_mapping_m8, id_mapping_m12, id_mapping_m1, id_mapping_m2])

runhist_array_f(runhist_array, id_mapping)


