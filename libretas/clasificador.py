# El de base
import numpy as np

# Las gráficas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Para separar los datos en entrenamiento y validación
from sklearn.model_selection import train_test_split

# Para normalizar los datos (desviación estandar)
from sklearn.preprocessing import StandardScaler

# Los conjuntos de datos artificiales típicos para probar clasificadores
from sklearn.datasets import make_moons, make_circles, make_classification

# Los métodos de aprendizaje a utilizar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

print("Cargado imports")
# %matplotlib inline

# Del modulo para general conjuntos de datos artificiales, para hacer conjuntos de lunas y circulos:
lunas = make_moons(noise=0.3, random_state=0)
circulos = make_circles(noise=0.2, factor=0.5, random_state=1)

# Ahora generamos un conjunto de datos linealmente separables
X, y = make_classification(n_features=2,             #  Dos dimensiones para poderlos graficar 
                           n_redundant=0,            #  Sin dimensiones redundantes, no nos interesa probar esto ahora
                           n_informative=2,          #  Las dos dimensiones informativas (no correlacionadas)
                           random_state=1,           #  Semilla, siempre la misma para que todos tengan los mismos resultados
                           n_clusters_per_class=1)   #  Una sola forma por clase para hacerlo más sencillo
print("X sin ruido: {}".format(X))

# Le agregamos ruido a la separación lineal (algunos puntos mal clasificados)
rng = np.random.RandomState(2)           #  Un generados de números pseudoaleatorios con la semilla impuesta

X += 2 * rng.uniform(size=X.shape)       
#  A cada punto se le suma un error con una distribución uniforme en ambas dimensiones

print("X con ruido: {}".format(X))

lineal = (X, y) # nuestro dataset de datos linealmente separables

datasets = [lunas, circulos, lineal]    # Una lista de tuplas (X, y)

for data in datasets: # quiero ver mis datos
    print("Dataset: {}".format(data))

# Y los graficamos para verlos

figure = plt.figure(figsize=(30, 10)) # matplotlib si dejas de utilizar una figura, close, para limpiar memoria
cm_escala = ListedColormap(['#FF0000', '#0000FF'])

print("Empezando a graficar")
for (i, ds) in enumerate(datasets):

    # Selecciona los valores del conjunto de datos y los escala
    X, y = ds
    X = StandardScaler().fit_transform(X) 

    # Grafica
    ax = plt.subplot(1, 3, i+1)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=150, cmap=cm_escala)
    ax.set_xlim(X[:, 0].min() - .5, X[:, 0].max() + .5)
    ax.set_ylim(X[:, 1].min() - .5, X[:, 1].max() + .5)
    ax.set_xticks(())
    ax.set_yticks(())
figure.subplots_adjust(left=.02, right=.98)    
plt.show()

print("Cargando metodos") 

titulos = [u"Vecinos próximos", "SVM lineal", "SVM gaussiano", u"Árbol de desición",
           u"Boseques aleatórios", "AdaBoost", "Naive Bayes", "Discriminante lineal",
           "Discriminante cuadrátco"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Colores brillantes
cm = plt.cm.RdBu
cm_escala = ListedColormap(['#FF0000', '#0000FF'])

print("KERNEL DANGER ZONE")
for (cual, ds) in enumerate(datasets):
    
    print("Base de datos {}".format(cual))
    figure = plt.figure(figsize=(30, 30))

    # Escalar y selecciona valores de entrenamiento y prueba
    X, y = ds
    X = StandardScaler().fit_transform(X)
    
    # Dividir el conjunto en un conjunto de entrenamiento y otro de aprendizaje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    # Meshgrid para clasificar toda la región y pintar las regiones
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - .5, X[:, 0].max() + .5, 0.02),
                         np.arange(X[:, 1].min() - .5, X[:, 1].max() + .5, 0.02))


    # Por cada clasificador
    for (i, (titulo, clf)) in enumerate(zip(titulos, classifiers)):
        
        # Escoge el subplot
        ax = plt.subplot(3, 3, i + 1)
        
        # El entrenamiento!!!!
        clf.fit(X_train, y_train)
        
        # Encuentra el error de validación
        score = clf.score(X_test, y_test)

        # Clasifica cada punto en el meshgrid
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Asigna un color a cada punto
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Grafica los datos de entrenamiento y prueba
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_escala, s=150)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_escala, s=150, alpha=0.6)


        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(titulo, size=30)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=30, horizontalalignment='right')

    figure.subplots_adjust(left=.02, right=.98)
    plt.show()
print("KERNEL DANGER ZONE PASS")
