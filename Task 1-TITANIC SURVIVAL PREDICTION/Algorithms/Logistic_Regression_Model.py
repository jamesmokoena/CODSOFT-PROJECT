
from sklearn.linear_model import LogisticRegression
from Titanic_program import titanic_train


Young_M= titanic_train['Name'].str.contains('Master', regex= True, na=False)
Young_F= titanic_train['Name'].str.contains('Miss', regex= True, na=False)
Adult_M= titanic_train['Name'].str.contains('Mr\.', regex= True, na=False)
Adult_F= titanic_train['Name'].str.contains('Mrs', regex= True, na=False)

print("Mean Age for Young Males: ",titanic_train.loc[Young_M,'Age'].mean())
print("Mean Age for Young Females: ",titanic_train.loc[Young_F,'Age'].mean())