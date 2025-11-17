import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
import os
import glob
import json

# Page config
st.set_page_config(page_title="Aksara Sasak AI")

@st.cache_resource
def load_model_simple():
    """Load model yang sudah trained"""
    return load_model('model/aksara_sasak_model.h5')

def predict_single_image(model, image):
    """Predict single image"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

def predict_batch(model, image_folder, aksara_names):
    """Predict semua image dalam folder + extract actual class dari parent folder"""
    results = []
    
    # Cari semua file gambar di SEMUA subfolder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(image_folder, "**", ext.upper()), recursive=True))
    
    for image_path in image_files:
        try:
            # Extract actual class dari parent folder name
            parent_folder = os.path.basename(os.path.dirname(image_path))
            
            # Load dan predict image
            image = Image.open(image_path)
            predictions = predict_single_image(model, image)
            
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_aksara = aksara_names[predicted_idx]
            
            # Simple check untuk correctness
            is_correct = parent_folder.lower() in predicted_aksara.lower()
            
            results.append({
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'actual_class': parent_folder,
                'predicted_class': predicted_aksara,
                'confidence': confidence,
                'is_correct': is_correct,
                'top_3_predictions': [
                    (aksara_names[i], predictions[i]) 
                    for i in np.argsort(predictions)[-3:][::-1]
                ]
            })
            
        except Exception as e:
            results.append({
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'actual_class': 'UNKNOWN',
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'is_correct': False,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def main():
    st.title("🖋️ Aksara Sasak Detector")
    st.write("Deteksi aksara Sasak dari gambar tunggal atau folder")
    
    # Load model
    try:
        model = load_model_simple()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.info("Pastikan file 'model/aksara_sasak_model.h5' ada")
        return

    # Load class indices
    try:
        with open('model/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        aksara_names = list(class_indices.keys())
    except Exception as e:
        st.error(f"❌ Gagal load class indices: {e}")
        return
    
    # Pilih mode
    mode = st.radio(
        "Pilih Mode Deteksi:",
        ["📷 Single Image", "📁 Batch Folder"],
        horizontal=True
    )
    
    if mode == "📷 Single Image":
        single_image_mode(model, aksara_names)
    else:
        batch_mode(model, aksara_names)

def single_image_mode(model, aksara_names):
    """Mode untuk single image prediction"""
    st.subheader("📷 Deteksi Single Image")
    
    uploaded_file = st.file_uploader(
        "Upload gambar aksara Sasak...", 
        type=['jpg', 'png', 'jpeg'],
        key="single"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_column_width=True)
        
        with col2:
            with st.spinner("🔍 Menganalisis aksara..."):
                predictions = predict_single_image(model, image)
                
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_aksara = aksara_names[predicted_idx]
            
            st.success(f"**Hasil:** {predicted_aksara}")
            st.info(f"**Tingkat Kepercayaan:** {confidence:.1%}")
            st.progress(float(confidence))
            
            st.subheader("🎯 Prediksi Lainnya:")
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            
            for idx in top_3_idx:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{aksara_names[idx]}**")
                with col_b:
                    st.write(f"{predictions[idx]:.1%}")
                st.progress(float(predictions[idx]))

def batch_mode(model, aksara_names):
    """Mode untuk batch prediction"""
    st.subheader("📁 Deteksi Batch Folder")
    
    # Input folder path
    folder_path = st.text_input(
        "Masukkan path folder yang berisi gambar:",
        placeholder="contoh: data/test_images atau C:/path/to/folder",
        key="folder_input"
    )
    
    if folder_path and os.path.exists(folder_path):
        if st.button("🚀 Mulai Batch Prediction", key="batch_predict"):
            with st.spinner("🔍 Menganalisis semua gambar dalam folder..."):
                results_df = predict_batch(model, folder_path, aksara_names)
            
            # Simpan ke session state agar tidak hilang saat rerun
            st.session_state.batch_results = results_df
            st.session_state.folder_path = folder_path
            
        # Tampilkan results jika ada di session state
        if 'batch_results' in st.session_state and not st.session_state.batch_results.empty:
            results_df = st.session_state.batch_results
            folder_path = st.session_state.folder_path
            
            st.subheader(f"📊 Hasil Batch Prediction ({len(results_df)} gambar)")
            
            # Summary statistics
            correct_predictions = results_df['is_correct'].sum()
            accuracy = correct_predictions / len(results_df) * 100
            avg_confidence = results_df['confidence'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Gambar", len(results_df))
            with col2:
                st.metric("Correct", f"{correct_predictions}")
            with col3:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Display results table
            st.dataframe(
                results_df[['file_name', 'actual_class', 'predicted_class', 'confidence', 'is_correct']],
                use_container_width=True
            )
            
            # Download results
            csv = results_df[['file_name', 'actual_class', 'predicted_class', 'confidence', 'is_correct']].to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv",
                key="download_csv"
            )
            
            # Detailed view untuk gambar tertentu
            st.subheader("🔍 Detail Per Gambar")
            selected_file = st.selectbox(
                "Pilih gambar untuk melihat detail:",
                results_df['file_name'].tolist(),
                key="file_selector"
            )
            
            if selected_file:
                selected_result = results_df[results_df['file_name'] == selected_file].iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Tampilkan gambar
                    try:
                        image = Image.open(selected_result['file_path'])
                        st.image(image, caption=selected_file, use_column_width=True)
                    except:
                        st.warning("Gagal menampilkan gambar")
                
                with col2:
                    # Tampilkan detail predictions
                    status = "✅ CORRECT" if selected_result['is_correct'] else "❌ WRONG"
                    st.write(f"**Status:** {status}")
                    st.write(f"**Actual Class:** {selected_result['actual_class']}")
                    st.write(f"**Predicted Class:** {selected_result['predicted_class']}")
                    st.write(f"**Confidence:** {selected_result['confidence']:.1%}")
                    
                    st.write("**Top 3 Predictions:**")
                    for class_name, score in selected_result['top_3_predictions']:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"- {class_name}")
                        with col_b:
                            st.write(f"{score:.1%}")
                        st.progress(float(score))
    
    elif folder_path:
        st.error("❌ Folder tidak ditemukan!")

if __name__ == "__main__":
    main()