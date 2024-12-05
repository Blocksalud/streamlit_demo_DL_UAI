import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# App Title
st.title("Simple Streamlit App")

# Input Text
user_input = st.text_input("Enter some text", "Hello, Streamlit!")

# Display user input
st.write("You entered:", user_input)

# Dropdown menu
options = st.selectbox(
    "Choose an option:",
    ["Option 1", "Option 2", "Option 3"]
)
st.write("You selected:", options)

# Slider
number = st.slider("Pick a number", 0, 100, 50)
st.write("You picked:", number)

# Generate and display random data
st.subheader("Random Data Visualization")
data = pd.DataFrame(
    np.random.randn(50, 2),
    columns=['X', 'Y']
)
st.line_chart(data)

# Plot with matplotlib
st.subheader("Custom Matplotlib Chart")
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Noisy Sine Wave")
plt.title("Sample Matplotlib Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
st.pyplot(plt)

# Checkbox to show/hide data
if st.checkbox("Show data table"):
    st.write(data)
