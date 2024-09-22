import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    y_true: labels (0 pour similaire, 1 pour dissemblable)
    y_pred: distances prédites entre les paires d'images
    margin: marge pour les paires dissemblables
    """
    # Conversion de y_true en float32 pour éviter des erreurs de type
    y_true = tf.cast(y_true, tf.float32)
    # Calcul des pertes pour les paires similaires et dissemblables
    loss_similar = (1 - y_true) * 0.5 * K.square(y_pred)
    loss_dissimilar = y_true * 0.5 * K.square(K.maximum(margin - y_pred, 0))
    return K.mean(loss_similar + loss_dissimilar)

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    
    # First Convolutional Block
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second Convolutional Block
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third Convolutional Block
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    
    return Model(input, x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def predict_similarity(img1_path, img2_path, model):
    img1 = load_img(img1_path, target_size=(128, 128), color_mode='grayscale')
    img2 = load_img(img2_path, target_size=(128, 128), color_mode='grayscale')
    img1 = img_to_array(img1) / 255.0  # Convertir en tableau NumPy et normaliser
    img2 = img_to_array(img2) / 255.0  # Convertir en tableau NumPy et normaliser
    img1 = img1.reshape(1, 128, 128, 1)  # Redimensionner pour correspondre à l'entrée du modèle
    img2 = img2.reshape(1, 128, 128, 1)  # Redimensionner pour correspondre à l'entrée du modèle
    similarity_score = model.predict([img1, img2], verbose=0)
    return similarity_score[0][0]

def classify_similarity_label(score, threshold=0.5):
    return "similar" if score > threshold else "dissimilar"

def classify_similarity(score, threshold=0.5):
    return 1 if score > threshold else 0

def model_init():
    input_shape = (128, 128, 1)
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Passer chaque entrée par le réseau de base
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    output = Dense(1, activation='sigmoid')(distance)
    model = Model([input_a, input_b], output)
    model.load_weights('./models/best_model_V6.keras')

    # Ajustement du learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def display_signature_comparison(img1_path, img1_1_path, img2_path, model):
    """
    Affiche les signatures et leurs comparaisons avec les scores de similarité.
    
    Arguments:
    img1_path -- Chemin vers la première signature
    img1_1_path -- Chemin vers la deuxième signature à comparer avec la première
    img2_path -- Chemin vers la troisième signature à comparer avec la première
    model -- Modèle pré-entrainé pour la prédiction de similarité
    """
    
    plt.figure(figsize=(12, 4))
    
    # Signature A
    plt.subplot(1, 3, 1)
    plt.imshow(load_img(img1_path, color_mode='grayscale'), cmap='gray')
    plt.title("Signature A")
    plt.axis('off')
    
    # Signature B
    plt.subplot(1, 3, 2)
    plt.imshow(load_img(img1_1_path, color_mode='grayscale'), cmap='gray')
    score_AB = predict_similarity(img1_path, img1_1_path, model)
    plt.title(f"Signature B\nClassification: {classify_similarity_label(score_AB)}\nScore: {score_AB:.2f}")
    plt.axis('off')
    
    # Signature C
    plt.subplot(1, 3, 3)
    plt.imshow(load_img(img2_path, color_mode='grayscale'), cmap='gray')
    score_AC = predict_similarity(img1_path, img2_path, model)
    plt.title(f"Signature C\nClassification: {classify_similarity_label(score_AC)}\nScore: {score_AC:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_performance(matrice):
    # Extraire les valeurs TN, FP, FN, TP de la matrice de confusion
    TN, FP, FN, TP = matrice[0][0], matrice[0][1], matrice[1][0], matrice[1][1]

    # Fonction pour calculer la précision (Accuracy)
    def calcul_precision(TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)

    # Fonction pour calculer la précision positive (Precision)
    def calcul_precision_positive(TP, FP):
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    # Fonction pour calculer le rappel (Recall)
    def calcul_rappel(TP, FN):
        return TP / (TP + FN) if (TP + FN) != 0 else 0

    # Fonction pour calculer le score F1 (F1-score)
    def calcul_f1_score(precision, rappel):
        return 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) != 0 else 0

    # Recalcul des métriques
    precision = calcul_precision(TP, TN, FP, FN)
    precision_positive = calcul_precision_positive(TP, FP)
    rappel = calcul_rappel(TP, FN)
    f1_score = calcul_f1_score(precision_positive, rappel)

    # Affichage des résultats
    print(f'Précision (Accuracy) : {precision:.4f}')
    print(f'Précision positive (Precision) : {precision_positive:.4f}')
    print(f'Rappel (Recall) : {rappel:.4f}')
    print(f'Score F1 : {f1_score:.4f}')