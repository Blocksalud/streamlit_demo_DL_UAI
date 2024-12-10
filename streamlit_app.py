import streamlit as st
import pickle
import numpy as np
import os
import json
from tensorflow.keras.models import load_model, model_from_json
from sklearn.exceptions import NotFittedError
from tensorflow.keras.utils import custom_object_scope


# Define the custom loss function
def weighted_focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

# Función para cargar modelos, incluyendo archivos .keras y SavedModel
def load_model_with_fix(model_path):
    try:
        # Check for .keras file
        if model_path.endswith(".keras") and os.path.exists(model_path):
            # Use custom_object_scope to register the loss function
            with custom_object_scope({'weighted_focal_loss': weighted_focal_loss()}):
                model = load_model(model_path)
            st.success(f"Modelo cargado exitosamente desde {model_path}")
            return model

        # Verificar si el modelo es una carpeta de SavedModel
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
            model = load_model(model_path)
            st.success(f"Modelo SavedModel cargado exitosamente desde {model_path}")
            return model

        # Verificar si el modelo está en formato JSON + H5
        config_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.weights.h5")
        if os.path.exists(config_path) and os.path.exists(weights_path):
            # Cargar y corregir config.json
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Corregir 'batch_shape' a 'input_shape' y 'dtype' si es necesario
            for layer in config.get("config", {}).get("layers", []):
                if "batch_shape" in layer["config"]:
                    layer["config"]["input_shape"] = layer["config"].pop("batch_shape")[1:]
                if isinstance(layer["config"].get("dtype"), dict):
                    layer["config"]["dtype"] = "float32"
            
            # Recrear el modelo desde JSON
            model = model_from_json(json.dumps(config))
            
            # Cargar los pesos con manejo de mismatches
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            st.success(f"Modelo JSON + H5 cargado exitosamente desde {model_path}")
            return model

        # Si no se encuentra ningún formato válido, lanzar error
        raise FileNotFoundError(f"No se encontraron archivos de modelo válidos en {model_path}.")
    except Exception as e:
        st.error(f"Error al cargar el modelo desde {model_path}: {e}")
        return None

# Función para cargar artefactos: scaler, PCA y encoders
def load_artifacts(model_path, scaler_path, pca_path, encoder_path):
    model = load_model_with_fix(model_path)
    
    try:
        # Cargar el StandardScaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            if not hasattr(scaler, "mean_"):
                raise NotFittedError("El StandardScaler no está ajustado.")
    except Exception as e:
        st.error(f"Error al cargar el scaler desde {scaler_path}: {e}")
        return None, None, None, None

    try:
        # Cargar PCA si existe
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    except FileNotFoundError:
        pca = None
        st.warning(f"No se encontró el archivo PCA en {pca_path}. Se omitirá PCA.")
    except Exception as e:
        st.error(f"Error al cargar PCA desde {pca_path}: {e}")
        return None, None, None, None

    try:
        # Cargar los LabelEncoders o mappings
        with open(encoder_path, "rb") as f:
            label_encoders = pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar los LabelEncoders desde {encoder_path}: {e}")
        return None, None, None, None

    return model, scaler, pca, label_encoders

# Rutas a los archivos de modelos y artefactos
model1_path = "model_1_sv"  # Cambiar a la ruta del archivo .keras o directorio SavedModel
scaler1_path = "scaler_corrected.pkl"
pca1_path = "pca_nw.pkl"  # Opcional, puede omitirse si no se usa PCA
encoder1_path = "label_encoders_nw.pkl"

model2_path = "model_2_sv"  # Cambiar a la ruta del archivo .keras o directorio SavedModel
scaler2_path = "scaler_corrected.pkl"
pca2_path = "pca_nw.pkl"  # Opcional, puede omitirse si no se usa PCA
encoder2_path = "label_encoders_nw.pkl"

model3_path = "best_model_hyper.keras"  # Cambiar a la ruta del archivo .keras o directorio SavedModel
scaler3_path = "scaler_corrected.pkl"
pca3_path = "pca_nw.pkl"  # Opcional, puede omitirse si no se usa PCA
encoder3_path = "label_encoders_nw.pkl"

# Cargar artefactos para ambos modelos
model1, scaler1, pca1, encoders1 = load_artifacts(model1_path, scaler1_path, pca1_path, encoder1_path)
model2, scaler2, pca2, encoders2 = load_artifacts(model2_path, scaler2_path, pca2_path, encoder2_path)
model3, scaler3, pca3, encoders3 = load_artifacts(model3_path, scaler3_path, pca3_path, encoder3_path)

# Verificar que al menos uno de los modelos se haya cargado correctamente
if not model1 and not model2 and not model3:
    st.error("No se pudieron cargar los modelos. Revisa las rutas y formatos.")
    st.stop()

# Título de la aplicación
st.title("Predicción con Múltiples Modelos")

# Selección del modelo a usar
model_option = st.selectbox(
    "Selecciona el modelo para usar",
    ("Modelo 1", "Modelo 2", "Modelo 3")
)

# Inicializar la lista de características de entrada
input_features = []

# Determinar qué encoders usar según el modelo seleccionado
if model_option == "Modelo 1" and encoders1:
    encoders = encoders1
elif model_option == "Modelo 2" and encoders2:
    encoders = encoders2
elif model_option == "Modelo 3" and encoders3:
    encoders = encoders3
else:
    encoders = None

# Entrada dinámica para columnas categóricas
if encoders:
    for col, encoder_info in encoders.items():
        if col.upper() != 'MONTO':  # Asumiendo que 'MONTO' es numérico
            if isinstance(encoder_info, dict):
                # Verificar si es un diccionario con 'mapping'
                if 'mapping' in encoder_info and isinstance(encoder_info['mapping'], dict):
                    mapping = encoder_info['mapping']
                    unique_values = list(mapping.keys())
                    selected_value = st.selectbox(f"Selecciona {col}", unique_values)
                    encoded_value = mapping[selected_value]
                # Verificar si contiene un 'encoder' con 'classes_'
                elif 'encoder' in encoder_info and hasattr(encoder_info['encoder'], 'classes_'):
                    encoder = encoder_info['encoder']
                    unique_values = list(encoder.classes_)
                    selected_value = st.selectbox(f"Selecciona {col}", unique_values)
                    encoded_value = encoder.transform([selected_value])[0]
                else:
                    # Si es un diccionario directo de mapeo
                    unique_values = list(encoder_info.keys())
                    selected_value = st.selectbox(f"Selecciona {col}", unique_values)
                    encoded_value = encoder_info[selected_value]
            elif hasattr(encoder_info, "classes_"):
                # Si encoder_info es una instancia de LabelEncoder
                unique_values = list(encoder_info.classes_)
                selected_value = st.selectbox(f"Selecciona {col}", unique_values)
                encoded_value = encoder_info.transform([selected_value])[0]
            elif isinstance(encoder_info, dict):
                # Si encoder_info es un diccionario de mapeo directo
                unique_values = list(encoder_info.keys())
                selected_value = st.selectbox(f"Selecciona {col}", unique_values)
                encoded_value = encoder_info[selected_value]
            else:
                st.warning(f"Tipo de encoder no soportado para la columna {col}: {type(encoder_info)}")
                continue

            st.write(f"{col}: {selected_value} → {encoded_value}")  # Salida de depuración
            input_features.append(encoded_value)

# Entrada numérica para 'MONTO'
monto = st.number_input("Introduce el Monto", value=0.0)
st.write(f"MONTO: {monto}")  # Salida de depuración
input_features.append(monto)

# Botón para realizar la predicción
if st.button("Predecir"):
    input_array = np.array([input_features])
    st.write(f"Entrada antes de escalar: {input_array}")  # Salida de depuración

    if model_option == "Modelo 1" and model1:
        try:
            # Transformar con StandardScaler
            input_scaled = scaler1.transform(input_array)
            st.write(f"Datos escalados (Modelo 1): {input_scaled}")  # Salida de depuración

            # Transformar con PCA si está disponible
            if pca1:
                input_pca = pca1.transform(input_scaled)
                prediction = model1.predict(input_pca)
            else:
                prediction = model1.predict(input_scaled)

            # Mostrar la predicción
            st.write(f"Predicción (Modelo 1): {prediction[0][0]:.4f}")
        except Exception as e:
            st.error(f"Error al procesar datos con el Modelo 1: {e}")

    elif model_option == "Modelo 2" and model2:
        try:
            # Transformar con StandardScaler
            if input_array.shape[1] != model2.input_shape[1]:
                raise ValueError(f"Input has {input_array.shape[1]} features, but model excepts {model2.input_shape[1]} features.")
            input_scaled = pca2.transform(input_array)
            st.write(f"Datos escalados (Modelo 2): {input_scaled}")  # Salida de depuración

            # Transformar con PCA si está disponible
            if pca2:
                input_pca = pca2.transform(input_scaled)
                prediction = model2.predict(input_pca)
            else:
                prediction = model2.predict(input_scaled)

            # Mostrar la predicción
            st.write(f"Predicción (Modelo 2): {prediction[0][0]:.4f}")
        except Exception as e:
            st.error(f"Error al procesar datos con el Modelo 2: {e}")

    elif model_option == "Modelo 3" and model3:
        try:
            # Ensure input shape matches model input
            if input_array.shape[1] != model3.input_shape[1]:
                raise ValueError(f"Input has {input_array.shape[1]} features, but model expects {model3.input_shape[1]} features.")

            # Transform input with scaler
            input_scaled = scaler3.transform(input_array)

            # Make prediction
            prediction = model3.predict(input_scaled)
            st.write(f"Prediction (Modelo 3): {prediction[0][0]:.4f}")

        except ValueError as ve:
            st.error(f"Feature mismatch error: {ve}")
        except Exception as e:
            st.error(f"Error processing data with Modelo 3: {e}")
    else:
        st.warning("No hay modelos o transformadores disponibles para la predicción.")
