a = [
    "r_sans_contexte.csv",
    "r_sans_contexte_caucasian.csv",
    "r_labels_corrected.csv",   
    "r_cultural_heritage.csv",
    "r_portrait_individual.csv",
    "r_feutures.csv",           
    "r_photo_of.csv",
    "g_sans.csv",        
    "g_formal.csv",      
    "g_picture_of.csv",  
    "g_wo_man.csv"
]

for i in a:
    input("\n")
    var_name = "df_" + i.replace(".csv", "")
    print(f"{var_name} = import_data('./scv_1/{i}')")
    if i[0] == "r":
        print(f"run_stats({var_name}, 'race')\n")
    elif i[0] == "g":
        print(f"run_stats({var_name}, 'gender')\n")
