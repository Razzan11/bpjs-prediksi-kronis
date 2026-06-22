import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Prediksi Penyakit Kronis", layout="wide")

@st.cache_resource
def load_model():
    with open('champion_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()
model = bundle['model']
le = bundle['label_encoder']
feature_cols = bundle['feature_cols']

st.title("🏥 Prediksi Penyakit Kronis Pasien BPJS")
st.write("Aplikasi ini mensimulasikan prediksi risiko 10 penyakit kronis berdasarkan demografi pasien.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📋 Panel Input")
    umur = st.slider("Umur", 0, 100, 55)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    gol_darah = st.selectbox("Golongan Darah", ["A", "B", "AB", "O"])
    
    predict_btn = st.button("Prediksi Penyakit Kronis", type="primary")

with col2:
    st.header("📊 Panel Hasil")
    if predict_btn:
        # Prepare input dict matching exact feature columns
        input_data = {col: 0 for col in feature_cols}
        input_data['Umur'] = umur
        
        if jenis_kelamin == "Perempuan":
            input_data['Jenis Kelamin_Perempuan'] = 1
            
        if gol_darah == "B":
            input_data['Golongan Darah_B'] = 1
        elif gol_darah == "AB":
            input_data['Golongan Darah_AB'] = 1
        elif gol_darah == "O":
            input_data['Golongan Darah_O'] = 1
            
        # Convert to DataFrame
        X_test = pd.DataFrame([input_data])[feature_cols]
        
        # Predict
        pred_idx = model.predict(X_test)[0]
        pred_proba = model.predict_proba(X_test)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        
        # Display Result
        st.success(f"Berdasarkan profil, pasien ini paling rentan terhadap: **{pred_label.upper()}** (Probabilitas: {max(pred_proba)*100:.1f}%)")
        
        st.subheader("Distribusi Probabilitas Semua Kelas")
        sorted_preds = sorted(zip(le.classes_, pred_proba), key=lambda x: -x[1])
        prob_df = pd.DataFrame(sorted_preds, columns=['Penyakit', 'Probabilitas'])
        prob_df['Persentase (%)'] = prob_df['Probabilitas'] * 100
        
        st.bar_chart(prob_df.set_index('Penyakit')['Persentase (%)'])
    else:
        st.info("Silakan isi data pasien di panel kiri dan klik tombol prediksi.")
