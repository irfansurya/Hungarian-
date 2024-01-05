#Import Library yang akan digunakan
import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

#Membaca dataset 
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

#melakukan pengecekan element (berjumlah 76),kemudian  menggabungkan 76 element dalam 10 kolom  menjadi 1 string/array yang sama
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

#Menghapus kolom terakhir dan pertama karena tidak digunakan 
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
#Mengubah typedata menjadi float
df = df.astype(float)
#Mengganti bilangan -9.0 menjadi null sesuai deskripsi
df.replace(-9.0, np.NaN, inplace=True)
#Memilih data yang akan digunakan sesuai dengan deskripsi dataset
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]
#Mapping nama kolom yang akan digunakan
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}
#Mengganti nama kolom
df_selected.rename(columns=column_mapping, inplace=True)
#Menghapus fitur data yang hampir 90% datanya memiliki nilai null dengan menggunakan fungsi drop
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

#Mengisi data yang terdapat nilai null  dengan menggunakan fungsi mean disetiap kolomnya
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

#Mengonversi typedata menjadi float
meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

#Membuat rata-rata dari datast dan kemudian membulatkannya
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

#Mengubah nilai null menjadi nilai mean yang telah dicari sebelumnya
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)
#Menghapus data duplikat
df_clean.drop_duplicates(inplace=True)

#Memisahkan fitur dengan target, lalu masukan kedalam variabel x dan y
X = df_clean.drop("target", axis=1)
y = df_clean['target']

#Melakukan Oversampling 
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

#Load Model
model = pickle.load(open("model/xgb_model.pkl", 'rb'))

#Melakukan Prediksi
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Penyakit Jantung",
  page_icon = ":hospital:"
)

# ========================================================================================================================================================================================

# ========================================================================================================================================================================================
st.title("Penyakit Jantung")

st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

#Sidebar
with tab1:
  st.sidebar.header("**Masukkan Informasi**")
   #Menginputkan usia dari pengguna dengan batas umur minimal dan maksimal dalam data set.
  age = st.sidebar.number_input(label=":blue[**Usia**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  #Menginputkan jenis kelamin dengan inisial Male = 1 dan Female = 0
  sex_sb = st.sidebar.selectbox(label=":blue[**Jenis Kelamin**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  #Menginputkan Data Chest Pain dalam bentuk dropdown
  cp_sb = st.sidebar.selectbox(label=":blue[**Jenis Nyeri Dada**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  #Menginputkan Resting blood pressure dariuser dengan batas Resting blood pressure minimal dalam dataset dan Resting blood pressure maksimal dalam dataset.
  trestbps = st.sidebar.number_input(label=":blue[**Tekanan Darah** (dalam mm Hg)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  #Menginputkan cholestoral dengan batas Resting blood pressure minimal dalam dataset dan cholestoral maksimal dalam dataset.
  chol = st.sidebar.number_input(label=":blue[**Kolesterol** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")
  
  #Meninputkan data dalam bentuk dropbox
  fbs_sb = st.sidebar.selectbox(label=":violet[**Gula Darah > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  #Meninputkan data dalam bentuk dropbox
  restecg_sb = st.sidebar.selectbox(label=":blue[**Hasil Elektrocardiografi**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  #Menginputkan maximum heart rate achieved dengan batas maximum heart rate achieved minimal dalam dataset dan maximum heart rate achieved maksimal dalam dataset.
  thalach = st.sidebar.number_input(label=":blue[**Denyut Jantung Maksimal Tercapai**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  #Menginputkan data dalam bentuk dropbox
  exang_sb = st.sidebar.selectbox(label=":blue[**Masuk Angin Akibat Olahraga?**]", options=["Tidak", "Iya"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "Tidak":
    exang = 0
  elif exang_sb == "Iya":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  #Menginputkan maximum heart rate achieved dengan batas maximum heart rate achieved minimal dalam dataset dan maximum heart rate achieved maksimal dalam dataset.
  oldpeak = st.sidebar.number_input(label=":blue[**Depresi disebabkan oleh olahraga dibandingkan istirahat**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  #Memasukan Data yang telah diinputkan dalam bentuk dataFrame
  data = {
    'Usia': age,
    'Jenis Kelamin': sex_sb,
    'Jenis Nyeri Dada': cp_sb,
    'Tekanan Darah': f"{trestbps} mm Hg",
    'Kolesterol': f"{chol} mg/dl",
    'Gula Darah > 120 mg/dl?': fbs_sb,
    'Hasil Elektrocardiografi': restecg_sb,
    'Denyut Jantung Maksimal Tercapai': thalach,
    'Masuk Angin Akibat Olahraga?': exang_sb,
    'Depresi': oldpeak,
  }

  preview_df = pd.DataFrame(data, index=['data'])

  st.header("Data Pasien")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  result = ":blue[-]"

  #Membuat tombol Prediksi dan fungsinya
  predict_btn = st.button("**Prediksi**", type="primary")
  
  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Sehat**]"
    elif prediction == 1:
      result = ":orange[**Penyakit jantung level 1**]"
    elif prediction == 2:
      result = ":orange[**Penyakit jantung level 2**]"
    elif prediction == 3:
      result = ":red[**Penyakit jantung level 3**]"
    elif prediction == 4:
      result = ":red[**Penyakit jantung level 4**]"

  st.write("")
  st.write("")
  st.subheader("Prediksi:")
  st.subheader(result)
  st.write(f"**_Akurasi Model_** :  :green[**{accuracy}**]% ")

#Membuat Tab 2 yaitu Multi predict
with tab2:
  st.header("Predict multiple data:")

  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV ", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Silahkan upload file CSV", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Sehat"
      elif prediction == 1:
        result = "Penyakit jantung level 1"
      elif prediction == 2:
        result = "Penyakit jantung level 2"
      elif prediction == 3:
        result = "Penyakit jantung level 3"
      elif prediction == 4:
        result = "Penyakit jantung level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Hasil Prediksi': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
