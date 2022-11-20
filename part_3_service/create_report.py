# pip install XlsxWriter
import os
import glob
import pandas as pd

PATH = os.path.join('part_3_service', 'predicted_images')

# List all files and directories in current directory
names_pred_list = sorted(os.listdir(PATH))
#print(names_pred_list)

col_start_rec = 0
row_start_rec = 0
with pd.ExcelWriter(os.path.join('part_3_service', 'results.xlsx'), engine='xlsxwriter') as writer:

    for fold_name in names_pred_list:
        csv_file = pd.read_csv(glob.glob(os.path.join(PATH, fold_name,'*.csv'))[0],
                               delimiter=';',
                               names=['filename', 'all_calc', 'reduce_calc'],
                               index_col=False
                              )
        
        if col_start_rec//15==1:
           col_start_rec = 0
           row_start_rec += len(csv_file)+3
        
        csv_file.to_excel(writer, 'Sheet1', 
                          startcol=col_start_rec, 
                          startrow=row_start_rec, 
                          header=['folder: '+ fold_name, 'all_calc', 'reduce_calc'],
                          index=False
                          )
        
        col_start_rec += 5
        
    
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    format_1 = workbook.add_format({'align': 'center'})
    format_2 = workbook.add_format({'border': 1})
    
    worksheet.set_column('A:O', 11, format_1) # Задаем ширину колонок с А по O 
    worksheet.conditional_format(0, 0,
                                 row_start_rec+len(csv_file)+3, 15,
                                 {'type': 'cell',
                                  'criteria': '!=',
                                  'value': '$ZZ$1',
                                  'format': format_2}
                                )

print('Well done!!!')