#importing the pacakages which are required 
import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.api as sm

#install plotly package 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

# =============================================================================
# #Mann-whitney test 
# data=pd.read_csv("C:/Users/suri/Desktop/practice programs/Hypothesis testing/with and without additive.csv")
# 
# #doing Normality test for Mann whitney
# #without additive Normality test
# withoutAdditive_data=stats.shapiro(data.Without_additive)
# withoutAdditive_pValue=withoutAdditive_data[1]
# print("p-value is: "+str(withoutAdditive_pValue))
# 
# #Additive normality test
# Additive=stats.shapiro(data.With_Additive)
# Additive_pValue=Additive[1]
# print("p-value is: "+str(Additive_pValue))
# 
# #Doing Mann-Whiteny test
# from scipy.stats import mannwhitneyu
# mannwhitneyu(data.Without_additive, data.With_Additive)
# 
# #############################End of Mann-whiteny test#####################################
# =============================================================================

# =============================================================================
# =============================================================================
# # ##Cutlet
# =============================================================================
# =============================================================================

#2- Sample T-Test
#Creditcard Cutlets data set 
Cutlets=pd.read_csv("D:/Training/ExcelR_2/Hypothesis_Testing/Cutlets.csv")
#Ho: There is no difference in the diameter of the cutlets 
#Ha: There is a difference in the diameter of the cutlets
#Doing Normality test 
#We consider Ho: Data are normal
#We consider Ha: Data are not normal

#renaming the label names to remove empty spaces
cols = Cutlets.columns
cols = cols.map(lambda x: x.replace(' ', '_'))
Cutlets.columns = cols

ACutlets=stats.shapiro(Cutlets.Unit_A)
ACutlets_pValue=ACutlets[1]
print("p-value is: "+str(ACutlets_pValue))


BCutlets=stats.shapiro(Cutlets.Unit_B)
BCutlets_pValue=BCutlets[1]
print("p-value is: "+str(BCutlets_pValue))
#we can proceed with the model 
#Varience test 
scipy.stats.levene(Cutlets.Unit_A, Cutlets.Unit_B)

#2 Sample T test 
#scipy.stats.ttest_ind(Cutlets.Unit_A,Cutlets.Unit_B)

scipy.stats.ttest_ind(Cutlets.Unit_A,Cutlets.Unit_B,equal_var = True)
###########################End of 2-Sample T-Test############################################

# =============================================================================
# =============================================================================
# # Laboratories Turn Around Time
# =============================================================================
# =============================================================================

#One way Anova
#Importing the data set of contractrenewal 
from statsmodels.formula.api import ols
cof=pd.read_csv("C:/Training/Analytics/Hypothesis_Testing/LabTAT.csv")
cof.columns="Laboratory1","Laboratory2","Laboratory3","Laboratory4"

#Normality test 
Lab1=stats.shapiro(cof.Laboratory1)    #Shapiro Test
Lab1_pValue=Lab1[1]
print("p-value is: "+str(Lab1_pValue))

Lab2=stats.shapiro(cof.Laboratory2)
Lab2_pValue=Lab2[1]
print("p-value is: "+str(Lab2_pValue))

Lab3=stats.shapiro(cof.Laboratory3)
Lab3_pValue=Lab3[1]
print("p-value is: "+str(Lab3_pValue))

Lab4=stats.shapiro(cof.Laboratory4)
Lab4_pValue=Lab4[1]
print("p-value is: "+str(Lab4_pValue))

#Varience Test 
scipy.stats.levene(cof.Laboratory1, cof.Laboratory2)
scipy.stats.levene(cof.Laboratory1, cof.Laboratory3)
scipy.stats.levene(cof.Laboratory1, cof.Laboratory4)

scipy.stats.levene(cof.Laboratory2, cof.Laboratory3)
scipy.stats.levene(cof.Laboratory2, cof.Laboratory4)

scipy.stats.levene(cof.Laboratory3, cof.Laboratory4)
#One-Way Anova

mod=ols('Laboratory1~Laboratory2+Laboratory3+Laboratory4',data=cof).fit()
aov_table=sm.stats.anova_lm(mod,type=2)
print(aov_table)
###########################End of One-Way Anova###################################################



# =============================================================================
# =============================================================================
# # Sales of products in four different regions
# =============================================================================
# =============================================================================
#Chi-Square test 
#Importing the data set of bahaman 
BuyerRatio=pd.read_csv("C:/Training/Analytics/Hypothesis_Testing/BuyerRatio.csv")
BuyerRatio.columns="Observed_values","East","West","North","South"
BuyerRatio = BuyerRatio.drop(['Observed_values'],axis=1)
print(BuyerRatio)

Chisquares_results=scipy.stats.chi2_contingency(BuyerRatio)
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue))

##########################End of chi-square test################################################


# =============================================================================
# =============================================================================
# # Call Center
# =============================================================================
# =============================================================================

callcenter=pd.read_csv("C:/Training/Analytics/Hypothesis_Testing/Costomer+OrderForm.csv")

#for index, row in callcenter.iterrows():
#    if row == 'Error_Free':
#        row.replace('Error_Free',1)
#print(row['Malta'])
#    
    
callcenter = callcenter.replace('Error Free','Error_Free')
callcenter = callcenter.replace('Error_Free',0)
callcenter = callcenter.replace('Defective',1)

callcenter.sum()

callcenter_test = pd.DataFrame(np.array([[271,267,269,280], [29, 33, 31,20]]), columns=['Phillipines', 'Indonesia', 'Malta','India'])

Chisquares_results=scipy.stats.chi2_contingency(callcenter_test)
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue))




#1 Sample Sign Test 
import statsmodels.stats.descriptivestats as sd
#importing the data set of signtest.csv
data=pd.read_csv("C:/Users/suri/Desktop/practice programs/Hypothesis testing/Signtest.csv")
#normality test 
data_socres=stats.shapiro(data.Scores)
data_pValue=data_socres[1]
print("p-value is: "+str(data_pValue))

#1 Sample Sign Test 
sd.sign_test(data.Scores,mu0=0)
############################End of 1 Sample Sign test###########################################

#2-Proportion Test 
two_prop_test=pd.read_csv("C:/Training/Analytics/Hypothesis_Testing/Costomer+OrderForm.csv")

count=pd.crosstab(two_prop_test["Weekdays"],two_prop_test["Weekend"])
count

#importing packages to do 2 proportion test
from statsmodels.stats.proportion import proportions_ztest
#we do the cross table and see How many adults or children are purchasing
tab = two_prop_test.groupby(['Error Free', 'Defective']).size()
count = np.array([58, 152]) #How many adults and childeren are purchasing
nobs = np.array([480, 740]) #Total number of adults and childern are there 

stat, pval = proportions_ztest(count, nobs,alternative='two-sided') 
#Alternative The alternative hypothesis can be either two-sided or one of the one- sided tests
#smaller means that the alternative hypothesis is prop < value
#larger means prop > value.
print('{0:0.3f}'.format(pval))
# two. sided -> means checking for equal proportions of Adults and children under purchased
# p-value = 6.261e-05 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

stat, pval = proportions_ztest(count, nobs,alternative='larger')
print('{0:0.3f}'.format(pval))
# Ha -> Proportions of Adults > Proportions of Children
# Ho -> Proportions of Children > Proportions of Adults
# p-value = 0.999 >0.05 accept null hypothesis 
# so proportion of Children > proportion of children 
# Do not launch the ice cream shop

###################################End of Two proportion test####################################




### Two Proportion test | Faltoons #################

n1 = 247
p1 = .37

n2 = 308
p2 = .39

population1 = np.random.binomial(1, p1, n1)
population2 = np.random.binomial(1, p2, n2)

sm.stats.ttest_ind(population1, population2)

>> (0.9309838177540464, 0.3522681761633615, 553.0)