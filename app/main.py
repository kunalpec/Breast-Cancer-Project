import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# prediction'
def add_pred_value(input_data):
    model=pickle.load(open('Breast_cancer.py/model.pkl','rb'))
    scaler=pickle.load(open('Breast_cancer.py/scaler.pkl','rb'))

    # convert input_data in array
    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaler=scaler.transform(input_array)
    # prediction model
    prediction=model.predict(input_array_scaler)

    st.subheader("Cell Cluster Prediction")
    st.write("The Cluster is :")
    if prediction[0]==0:
        st.write("**Result:** Benign")
    else:
        st.write("**Result:** Malicious")
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaler)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaler)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# Clean data
def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True, errors='ignore')
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    data = data.dropna()  # Drop NaN values
    print(data.head(5))
    return data

# Scale 
def scale_inputs(input_dict):
    scaled_dict = {}
    data=get_clean_data()
    X=data.drop(["diagnosis"],axis=1)
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        # scaled_dict[key] = scaled_value
    return scaled_dict

# Add side bar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Scaling")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave Points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal Dimension (se)", "fractal_dimension_se"),
        ("Radius (Worst)", "radius_worst"),
        ("Texture (Worst)", "texture_worst"),
        ("Perimeter (Worst)", "perimeter_worst"),
        ("Area (Worst)", "area_worst"),
        ("Smoothness (Worst)", "smoothness_worst"),
        ("Compactness (Worst)", "compactness_worst"),
        ("Concavity (Worst)", "concavity_worst"),
        ("Concave Points (Worst)", "concave points_worst"),
        ("Symmetry (Worst)", "symmetry_worst"),
        ("Fractal Dimension (Worst)", "fractal_dimension_worst"),

    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

# Radar chart
def get_radar_char(input_data):

    scale_inputs(input_data)

    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area',
        'Smoothness', 'Compactness', 'Concavity',
        'Concave Points', 'Symmetry', 'Fractal Dimension'
    ]
    # Add logic here for radar chart generation

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0, 1],gridcolor='black')),
        showlegend=True
        
    )
    return fig


# Main app
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    input_data = add_sidebar()

    # Main content
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("""
        Welcome to the **Breast Cancer Predictor** tool. This app uses machine learning models to predict 
        whether a breast cancer tumor is malignant or benign based on input data. Stay healthy and informed. 
        """)

    # columns
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_char = get_radar_char(input_data)
        st.plotly_chart(radar_char)

    with col2:
        add_pred_value(input_data)

if __name__ == "__main__":
    main()
