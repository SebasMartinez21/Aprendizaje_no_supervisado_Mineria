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
# ⚜️ Método del codo
# ========================================================================================================
inertias = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    inertias.append(kmeans.inertia_)

plt. figure(figsize=(6,4))
plt.plot(k_range, inertias, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo Segmentación de jugadores")
plt.show()

