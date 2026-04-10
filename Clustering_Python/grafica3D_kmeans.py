import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# ========================================================================================================
# ⚜️ Carga de datos 
# ========================================================================================================
df = pd.read_csv("C:\\Users\\sebas\\OneDrive\\Desktop\\_SemestresCursados_\\5to_Semestre\\Mineria_Datos\\Talleres60x\\Aprendizaje_no_supervisado_Mineria\\Codigo_Datasets\\dataset_clustering.csv")
print(df)

# ========================================================================================================
# ⚜️ Redondear antes de convertir a int (por si acaso hay decimales)
# ========================================================================================================
df["MP"] = df["MP"].round().astype(int)

df["Gls"] = df["Gls"].round().astype(int)

df["Ast"] = df["Ast"].round().astype(int)
print("\nRedondeando: \n", df)

x = df[["MP", "Gls", "Ast"]]

# ========================================================================================================
# ⚜️ Entrenar K-Means con k=4 (Por el método del codo) 
# ========================================================================================================
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(x)

df["Cluster"] = kmeans.labels_

# ========================================================================================================
# ⚜️ Graficar en 3D sin anotaciones (labels)
# ========================================================================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Colocar colores a los clusters
sc = ax.scatter(df["MP"], df["Gls"], df["Ast"], 
                c=df["Cluster"], cmap="viridis", s=50)

# Etiquetas de ejes
ax.set_xlabel("Partidos Jugados (MP)")
ax.set_ylabel("Goles (Gls)")
ax.set_zlabel("Asistencias (Ast)")
plt.title("Segmentación de jugadores Kmeans, k=4")


plt.show()
