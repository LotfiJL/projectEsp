import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#--------------Question 1 :

#--------------Charger la base de données initiale :

df_initial = pd.read_excel('base_initiale.xlsx' )  

#--------------Afficher les premières lignes :

#print(df_initial.head())  


#--------------Question 2 :

#--------------Créer des DataFrames vides pour chaque continent :

df_asia = pd.DataFrame(columns=df_initial.columns)  # DataFrame pour l'Asie
df_europe = pd.DataFrame(columns=df_initial.columns)  # DataFrame pour l'Europe
df_africa = pd.DataFrame(columns=df_initial.columns)  # DataFrame pour l'Afrique
df_americas_oceania = pd.DataFrame(columns=df_initial.columns)  # DataFrame pour les Amériques et l'Océanie


#--------------Afficher que les Dataframe sont bien créées :

#print(df_asia)
#print(df_europe)
#print(df_africa)
#print(df_americas_oceania)



#--------------Question 3 :


def classify_by_continent(df):
    global df_asia, df_europe, df_africa, df_americas_oceania  # Déclarer les DataFrames comme globaux

    #--------------parcourir chaque ligne du DataFrame initial :
    for index, row in df.iterrows():  #iterrows() permet d'itérer sur chaque ligne du DataFrame df.

        #--------------récupèrer le nom du continent pour chaque pays :       
        continent = row['Continent']

        #--------------ajouter le pays au DataFrame correspondant :
        if continent == 'Asia':
            df_asia = pd.concat([df_asia, pd.DataFrame([row])], ignore_index=True)
        elif continent == 'Europe':
            df_europe = pd.concat([df_europe, pd.DataFrame([row])], ignore_index=True)
        elif continent == 'Africa':
            df_africa = pd.concat([df_africa, pd.DataFrame([row])], ignore_index=True)
        else:
            df_americas_oceania = pd.concat([df_americas_oceania, pd.DataFrame([row])], ignore_index=True)
    
    #--------------Retourner les DataFrames mis à jour :
    return df_asia, df_europe, df_africa, df_americas_oceania

#--------------Appel de la fonction pour classifier les pays selon leur continent :
df_asia, df_europe, df_africa, df_americas_oceania = classify_by_continent(df_initial)

#--------------Afficher les résultats pour chaque DataFrame :
#print("Asia DataFrame:")
#print(df_asia.head())

#print("\nEurope DataFrame:")
#print(df_europe.head())

#print("\nAfrica DataFrame:")
#print(df_africa.head())

#print("\nAmericas and Oceania DataFrame:")
#print(df_americas_oceania.head())



#---------question 5 : 

#--------------Suppression de la colonne 'GDP' et 'continent' dans chaque dataframe : autre methode

def clean_dataframe(df):
    if 'GDP' in df.columns:
        df.drop('GDP', axis=1, inplace=True)  # Suppression de la colonne GDP //axis =1 pour dire supprimer les col 
    if 'continent' in df.columns:
        df.drop('continent', axis=1, inplace=True)  # Suppression de la colonne continent ////inplace =true :l'opération est effectuée directement sur le DataFrame ou la série d'origine
    return df

#--------------Appliquer cette fonction à chaque dataframe :

df_asia = clean_dataframe(df_asia)
df_europe = clean_dataframe(df_europe)
df_africa = clean_dataframe(df_africa)
df_americas_oceania = clean_dataframe(df_americas_oceania)

#--------------Affichage pour vérifier les DataFrames nettoyés :

#print("Asia sans GDP et continent:")
#print(df_asia.head())

#print("\nEurope sans GDP et continent:")
#print(df_europe.head())

#print("\nAfrica sans GDP et continent:")
#print(df_africa.head())

#print("\nAmericas and Oceania sans GDP et continent:")
#print(df_americas_oceania.head())



#--------------question 8 :

def replace_missing_values(df):
    """
    Remplace les valeurs manquantes dans le dataframe par la moyenne de chaque colonne numérique.
    """
    #--------------Liste des colonnes numériques :

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    #-------------- Remplacement des valeurs manquantes par la moyenne pour chaque colonne numérique :
    for column in numeric_columns:
        mean_value = df[column].mean()  # Calcul de la moyenne de la colonne
        df[column].fillna(mean_value)  # fillna(value) : Remplace toutes les valeurs manquantes (NaN) dans la colonne avec la valeur spécifiée (ici, mean_value) // inplace=True : Modifie la colonne directement dans le DataFrame sans créer une copie
    
    return df
#--------------Nettoyage des valeurs manquantes pour chaque dataframe :
df_asia = replace_missing_values(df_asia)
df_europe = replace_missing_values(df_europe)
df_africa = replace_missing_values(df_africa)
df_americas_oceania = replace_missing_values(df_americas_oceania)

#--------------Vérification après nettoyage :

#print("Asia DataFrame after cleaning:")
#print(df_asia.head())

#print("\nEurope DataFrame after cleaning:")
#print(df_europe.head())

#print("\nAfrica DataFrame after cleaning:")
#print(df_africa.head())

#print("\nAmericas and Oceania DataFrame after cleaning:")
#print(df_americas_oceania.head())


#----question 9 :


def assign_economic_growth_level(row):
    if row['GDP growth'] > 0:
        return 'High'
    elif row['GDP growth'] >= -2:
        return 'Medium'
    else:
        return 'Low'
#--------------Conversion de la colonne 'GDP growth' en numérique : garantir que toutes les valeurs dans GDP growth sont prêtes pour des opérations numériques comme des comparaisons ou des calculs.


df_asia['GDP growth'] = pd.to_numeric(df_asia['GDP growth'], errors='coerce') #errors='coerce' : Si une valeur ne peut pas être convertie, elle sera remplacée par NaN (valeur manquante
df_europe['GDP growth'] = pd.to_numeric(df_europe['GDP growth'], errors='coerce')
df_africa['GDP growth'] = pd.to_numeric(df_africa['GDP growth'], errors='coerce')
df_americas_oceania['GDP growth'] = pd.to_numeric(df_americas_oceania['GDP growth'], errors='coerce')

#--------------Vérification que la conversion a bien eu lieu :
#print(df_asia[['Country', 'GDP growth']].head()) #Afficher les  premières lignes de Country et GDP growth pour vérifier que la conversion a bien eu lieu.

#--------------Ajouter une nouvelle colonne Economic growth level à chaque DataFrame en appliquant la fonction assign_economic_growth_level pour chaque Df :

df_asia['Economic growth level'] = df_asia.apply(assign_economic_growth_level, axis=1) #axis=1 : lignes , axis=0 : colonnes
df_europe['Economic growth level'] = df_europe.apply(assign_economic_growth_level, axis=1)
df_africa['Economic growth level'] = df_africa.apply(assign_economic_growth_level, axis=1)
df_americas_oceania['Economic growth level'] = df_americas_oceania.apply(assign_economic_growth_level, axis=1)

#--------------Vérification des résultats :

#print("Asia DataFrame after adding Economic growth level:")
#print(df_asia[['Country', 'GDP growth', 'Economic growth level']].head())

#print("\nEurope DataFrame after adding Economic growth level:")
#print(df_europe[['Country', 'GDP growth', 'Economic growth level']].head())

#print("\nAfrica DataFrame after adding Economic growth level:")
#print(df_africa[['Country', 'GDP growth', 'Economic growth level']].head())

#print("\nAmericas and Oceania DataFrame after adding Economic growth level:")
#print(df_americas_oceania[['Country', 'GDP growth', 'Economic growth level']].head())



#--------------------------------------------------Part2-------------------------------------
#question 1 :


import math

# Détermination du nombre d'observations dans la dataframe initiale : nombre de lignes
n_observations = len(df_initial)
#print(f"n_observations : {n_observations}")
# Calcul du nombre de classes en utilisant la règle de la racine carrée
k_classes = math.ceil(math.sqrt(n_observations)) #math.ceil arrondit toujours au nombre entier supérieur  

#print(f"Nombre de classes pour la dataframe d'origine (règle de la racine carrée) : {k_classes}")


#q2

def determine_classes(df, variable, k_classes):
    # Calculer la plage des données
    min_value = df[variable].min()  # Valeur minimale de la variable
    max_value = df[variable].max()  # Valeur maximale de la variable
    print("lotfi",max_value)
    # Calculer la largeur de chaque classe
    class_width = (max_value - min_value) / k_classes
    
    # Créer les bornes de chaque classe
    class_intervals = []
    for i in range(k_classes): #classification par intervalle 
        lower_bound = min_value + i * class_width
        upper_bound = min_value + (i + 1) * class_width
        class_intervals.append((lower_bound, upper_bound))
    # Renvoi de la liste des intervalles (classes) calculés
    return class_intervals

# Exemple d'utilisation pour une variable 'GDP'
k_classes = math.ceil(math.sqrt(len(df_initial)))  # Utilisation de la règle de la racine carrée
class_intervals = determine_classes(df_initial, 'GDP', k_classes)
#print("Classes déterminées :")
#Affich de maniére clair l'ouptut
for interval in class_intervals:    # affiche l'intervalle de la classe sous forme de chaîne de caractères.
    print(f"{interval[0]:.2f} - {interval[1]:.2f}") #:.2f : Le formatage :.2f arrondit les valeurs à deux décimales pour un affichage claire : Si l'intervalle est (0, 100), la ligne affichera 0.00 - 100.00
 


    #q3 : calculer les effectifs relatifs et les fréquences relatives pour chaque classe 
    def calculate_class_frequencies(df, variable, class_intervals): #variable : la variable à analyser (par exemple, le PIB ou GDP).
        # Créer une liste pour stocker les effectifs relatifs de chaque classe
        class_frequencies = []

        # Total des observations
        n_observations = len(df)

        # Pour chaque intervalle de classe, compter le nombre d'observations
        for interval in class_intervals:
            # Filtrer les observations qui sont dans l'intervalle
            lower_bound, upper_bound = interval
            count_in_class = len(df[(df[variable] >= lower_bound) & (df[variable] < upper_bound)]) #df[(df : filters the DataFrame df based on the boolean Series
    
            # Calcul de l'effectif relatif
            relative_frequency = count_in_class / n_observations
            if count_in_class > 0:  # Filtrer les classes avec effectif nul
                class_frequencies.append((interval, count_in_class, relative_frequency))

        return class_frequencies

    # Exemple d'utilisation pour une variable 'GDP'
    class_frequencies = calculate_class_frequencies(df_initial, 'GDP', class_intervals)

    # Afficher les résultats
    print("Effectifs relatifs et fréquences relatives pour chaque classe :")
    for interval, count, relative_freq in class_frequencies:
        print(f"{interval[0]:.2f} - {interval[1]:.2f} | Effectif: {count} | Fréquence relative: {relative_freq:.4f}")

    # Préparer les données pour le graphique
    class_labels = [f"{interval[0]:.2f} - {interval[1]:.2f}" for interval, _, _ in class_frequencies] #(_) are placeholders to ignore the other two elements (count and relative_frequency) because we don't need them here.
    effectifs_rel = [relative_freq for _, _, relative_freq in class_frequencies] #_) are used as placeholders because we don’t need the interval and count values, just the relative_frequency

    # Barplot pour les effectifs relatifs
    #plt.figure(figsize=(10, 6))
    #plt.bar(class_labels, effectifs_rel, color='skyblue')
    #plt.title('Effectifs relatifs par classe')
    #plt.xlabel('Classes')
    #plt.ylabel('Effectif relatif')
    #plt.xticks(rotation=45, ha='right')
    #plt.tight_layout()
    #plt.show()

    # Diagramme circulaire pour les fréquences relatives
    frequencies_rel = effectifs_rel  # Les fréquences relatives sont déjà calculées

    #plt.figure(figsize=(8, 8))
    #plt.pie(frequencies_rel, labels=class_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    #plt.title('Répartition des fréquences relatives par classe')
    #plt.axis('equal')  # Pour rendre le cercle circulaire
    #plt.show()

    # Identifier la classe dominante
classe_dominante = max(class_frequencies, key=lambda x: x[2])  # Basé sur la fréquence relative
#print(f"Classe dominante : {classe_dominante[0]} avec une fréquence relative de {classe_dominante[2]:.4f}")

# Vérifier la distribution totale
frequence_totale = sum([freq for _, _, freq in class_frequencies])
#print(f"Somme des fréquences relatives : {frequence_totale:.4f} (devrait être proche de 1.0)")



def calculate_class_centers_and_cumulative(df, variable, class_intervals):
    centers = []
    cumulative_freqs = []
    total = 0
    n = len(df)
    
    for interval in class_intervals:
        lower, upper = interval
        center = (lower + upper) / 2
        count = len(df[(df[variable] >= lower) & (df[variable] < upper)])
        relative_freq = count / n
        total += relative_freq
        
        centers.append(center)
        cumulative_freqs.append(total)
    
    return centers, cumulative_freqs

centers, cumulative_freqs = calculate_class_centers_and_cumulative(df_initial, 'GDP', class_intervals)

# Affichage des résultats
print("Centres des classes :", centers)
print("Fréquences cumulées relatives :", cumulative_freqs)







def weighted_mean(centers, relative_freqs):
    # La moyenne pondérée est la somme des produits de chaque centre de classe et sa fréquence relative
    return sum(center * freq for center, freq in zip(centers, relative_freqs))


def calculate_median(cumulative_freqs, centers, class_intervals):
    # Identifier l'intervalle où la médiane se situe
    for i, cumulative_freq in enumerate(cumulative_freqs):
        if cumulative_freq >= 0.5:
            lower, upper = class_intervals[i]
            center = centers[i]
            return center
    return None
def calculate_mode(df, variable, class_intervals):
    counts = [len(df[(df[variable] >= lower) & (df[variable] < upper)]) for lower, upper in class_intervals]
    max_count_index = counts.index(max(counts))
    return class_intervals[max_count_index]


def calculate_quartiles(cumulative_freqs, class_intervals, q):
    for i, cumulative_freq in enumerate(cumulative_freqs):
        if cumulative_freq >= q:
            lower, upper = class_intervals[i]
            return (lower + upper) / 2
    return None



def plot_barplot(class_labels, relative_freqs):
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, relative_freqs, color='skyblue')
    plt.title('Effectifs relatifs par classe')
    plt.xlabel('Classes')
    plt.ylabel('Effectif relatif')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_pie_chart(class_labels, relative_freqs):
    plt.figure(figsize=(8, 8))
    plt.pie(relative_freqs, labels=class_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Répartition des fréquences relatives par classe')
    plt.axis('equal')  # Pour rendre le cercle circulaire
    plt.show()



def calculate_variance(centers, relative_freqs, weighted_mean):
    return sum(((center - weighted_mean) ** 2) * freq for center, freq in zip(centers, relative_freqs))


def calculate_std_deviation(variance):
    return variance ** 0.5


def calculate_cv(mean, std_deviation):
    return (std_deviation / mean) * 100



def calculate_skewness(df, variable):
    return stats.skew(df[variable])


def calculate_kurtosis(df, variable):
    return stats.kurtosis(df[variable])


weighted_mean_val = weighted_mean(centers, effectifs_rel)
print(f"Moyenne pondérée : {weighted_mean_val}")

median = calculate_median(cumulative_freqs, centers, class_intervals)
print(f"Médiane : {median}")

mode = calculate_mode(df_initial, 'GDP', class_intervals)
print(f"Mode : {mode}")

first_quartile = calculate_quartiles(cumulative_freqs, class_intervals, 0.25)
print(f"Premier quartile : {first_quartile}")

third_quartile = calculate_quartiles(cumulative_freqs, class_intervals, 0.75)
print(f"Troisième quartile : {third_quartile}")


variance_val = calculate_variance(centers, effectifs_rel, weighted_mean_val)
std_deviation = calculate_std_deviation(variance_val)
cv = calculate_cv(weighted_mean_val, std_deviation)

print(f"Variance : {variance_val}")
print(f"Écart-type : {std_deviation}")
print(f"Coefficient de variation : {cv}")


skewness = calculate_skewness(df_initial, 'GDP')
kurtosis = calculate_kurtosis(df_initial, 'GDP')

print(f"Asymétrie : {skewness}")
print(f"Kurtose : {kurtosis}")


plot_barplot(class_labels, effectifs_rel)
plot_pie_chart(class_labels, effectifs_rel)
plt.figure(figsize=(10, 6))
df_initial.boxplot(column='GDP')
plt.title('Boxplot for GDP in df_initial')
plt.show()
