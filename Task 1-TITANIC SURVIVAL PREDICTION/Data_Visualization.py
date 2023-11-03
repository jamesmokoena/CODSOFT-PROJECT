from Titanic_program import titanic_train
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def PclasS_VS_Survival():
    titanic_train.Pclass.value_counts()
    titanic_train.groupby('Pclass').Survived.value_counts()
    sns.barplot(x='Pclass', y='Survived', data=titanic_train)
    plt.show()

def Sex_vs_Survival(): 
    titanic_train.groupby('Sex').Survived.value_counts()
    titanic_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
    sns.barplot(x='Sex', y='Survived', data=titanic_train)
    plt.show()
    

def Pclass_and_Sex_vs_Survival():
    tab = pd.crosstab(titanic_train['Pclass'], titanic_train['Sex'])
    print (tab)
    tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
    plt.xlabel('Pclass')
    plt.ylabel('Percentage')
    plt.show()

def Pclass_Sex_and_Embarked_vs_Survival():
    sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=titanic_train)  
    plt.show()

def Embarked_vs_Survived():
    titanic_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=titanic_train)
    plt.show()



def Parch_vs_Survival():
    titanic_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
    sns.barplot(x='Parch', y='Survived', ci=None, data=titanic_train) # ci=None will hide the error bar
    plt.show()


def Age_vs_Survival():
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    sns.violinplot(x="Embarked", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax1)
    sns.violinplot(x="Pclass", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax2)
    sns.violinplot(x="Sex", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax3)
    plt.show()


def Diplayer():
    print("\n  Which Relationship between Features and Survival do you want look at:\n \n 1. PclasS_VS_Survival\n 2. Sex_vs_Survival\n\
 3. Pclass_and_Sex_vs_Survival\n 4. Pclass_Sex_and_Embarked_vs_Survival\n 5. Embarked_vs_Survived\n 6. Parch_vs_Survival\n\
 7. Age_vs_Survival\n")
    answer = int(input ("SELECTION: (e.g 2):  \n"))

    if answer == 1:
        PclasS_VS_Survival()
    elif answer ==2 :
        Sex_vs_Survival()

    elif answer ==3 :
        Pclass_and_Sex_vs_Survival()

    elif answer ==4 :
        Pclass_Sex_and_Embarked_vs_Survival()

    elif answer ==5 :
        Embarked_vs_Survived()

    elif answer ==6 :
        Parch_vs_Survival()

    elif answer ==7 :
        Age_vs_Survival()   



if __name__ == '__main__':
    Diplayer()
                 
   
               