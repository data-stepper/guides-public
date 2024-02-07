# Why you shouldn't use Principal Component Regression

Principal Component Regression (PCR) is a common technique of preventing overfitting by reducing the number of features in a (linear) regression model.
However, in this guide I will show you when PCR fails to capture important information and how you need to analyze your features to avoid this.

To formulate our simple regression problem, let $Y \in \mathbb{R}^n$ be our target variable vector (we have $n \in \mathbb{N}$ samples available).
Let $X \in \mathbb{R}^{n \times p}$ be our feature matrix (we have $p \in \mathbb{N}$ features available).

In the classical linear regression model, we want to find a coefficient vector $\beta \in \mathbb{R}^p$ and an intercept $\alpha \in \mathbb{R}$ 
such that we can estimate our target variable $Y$ by the following equation:
$$\hat{Y} = X \beta + \alpha$$

Now if we have lots of features available, this leads to overfitting as the model has *too many degrees of freedom*.
Therefore, one may want to reduce the number of features picking only important ones that are, at best, uncorrelated so our linear regression is efficient.

PCA does this nicely for us by finding the principal components of our feature matrix $X$.
These are the vectors in $\mathbb{R}^p$ that capture the most variance in our data.

One may suspect that these also explain $Y$ best, but, as you'll see in this guide, this is where the problem lies.


```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS

np.random.seed(123)

# Let's create some dummy variables first
n: int = 10_000
p: int = 10
std_uninformative: float = 10.0
std_informative: float = 0.1

# This is our dependent variable, which we want to predict
y = np.random.standard_normal(size=(n, 1))

# Now we fill our feature matrix with random, high variance but uninformative features
# and finally one low variance, but perfectly informative feature
X = np.random.standard_normal((n, p - 1)) * std_uninformative
X = np.hstack((X, y * std_informative))

X.shape, y.shape
```




    ((10000, 10), (10000, 1))



We have $n=10,000$ samples and $p=10$ features in total, $9$ of which are complete noise and the last one is just our scaled $Y$.


```python
OLS(y, X).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>       <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>           <td>2.802e+30</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 07 Feb 2024</td> <th>  Prob (F-statistic):</th>            <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:53:23</td>     <th>  Log-Likelihood:    </th>          <td>3.0184e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 10000</td>      <th>  AIC:               </th>          <td>-6.037e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  9990</td>      <th>  BIC:               </th>          <td>-6.036e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>               <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>               <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th>  <td> 1.821e-16</td> <td> 1.88e-17</td> <td>    9.690</td> <td> 0.000</td> <td> 1.45e-16</td> <td> 2.19e-16</td>
</tr>
<tr>
  <th>x2</th>  <td> 1.066e-15</td> <td> 1.88e-17</td> <td>   56.647</td> <td> 0.000</td> <td> 1.03e-15</td> <td>  1.1e-15</td>
</tr>
<tr>
  <th>x3</th>  <td>  9.33e-16</td> <td>  1.9e-17</td> <td>   49.207</td> <td> 0.000</td> <td> 8.96e-16</td> <td>  9.7e-16</td>
</tr>
<tr>
  <th>x4</th>  <td> 2.758e-16</td> <td> 1.88e-17</td> <td>   14.639</td> <td> 0.000</td> <td> 2.39e-16</td> <td> 3.13e-16</td>
</tr>
<tr>
  <th>x5</th>  <td> 6.278e-16</td> <td> 1.91e-17</td> <td>   32.951</td> <td> 0.000</td> <td>  5.9e-16</td> <td> 6.65e-16</td>
</tr>
<tr>
  <th>x6</th>  <td> 6.743e-16</td> <td> 1.89e-17</td> <td>   35.661</td> <td> 0.000</td> <td> 6.37e-16</td> <td> 7.11e-16</td>
</tr>
<tr>
  <th>x7</th>  <td> 2.783e-16</td> <td> 1.88e-17</td> <td>   14.793</td> <td> 0.000</td> <td> 2.41e-16</td> <td> 3.15e-16</td>
</tr>
<tr>
  <th>x8</th>  <td> 3.055e-16</td> <td>  1.9e-17</td> <td>   16.103</td> <td> 0.000</td> <td> 2.68e-16</td> <td> 3.43e-16</td>
</tr>
<tr>
  <th>x9</th>  <td>-3.698e-16</td> <td> 1.87e-17</td> <td>  -19.812</td> <td> 0.000</td> <td>-4.06e-16</td> <td>-3.33e-16</td>
</tr>
<tr>
  <th>x10</th> <td>   10.0000</td> <td> 1.89e-15</td> <td> 5.29e+15</td> <td> 0.000</td> <td>   10.000</td> <td>   10.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.204</td> <th>  Durbin-Watson:     </th> <td>   2.028</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.202</td> <th>  Jarque-Bera (JB):  </th> <td>   3.178</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.035</td> <th>  Prob(JB):          </th> <td>   0.204</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.052</td> <th>  Cond. No.          </th> <td>    102.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



We can see that our standard OLS regression (with all features) solves the problem perfectly as we get an $R^2$ of 1.0.
Also, all the coefficients are negligible, except for the last one, which is the only informative feature.

This is great!


```python
X_pc = PCA(n_components=3).fit_transform(X)

OLS(y, X_pc).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>  -0.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>  0.3306</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 07 Feb 2024</td> <th>  Prob (F-statistic):</th>           <td> 0.803</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:53:23</td>     <th>  Log-Likelihood:    </th>          <td> -14170.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 10000</td>      <th>  AIC:               </th>          <td>2.835e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  9997</td>      <th>  BIC:               </th>          <td>2.837e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>    0.0002</td> <td>    0.001</td> <td>    0.238</td> <td> 0.812</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>x2</th> <td>   -0.0009</td> <td>    0.001</td> <td>   -0.933</td> <td> 0.351</td> <td>   -0.003</td> <td>    0.001</td>
</tr>
<tr>
  <th>x3</th> <td>   -0.0003</td> <td>    0.001</td> <td>   -0.255</td> <td> 0.799</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.389</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.823</td> <th>  Jarque-Bera (JB):  </th> <td>   0.421</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.004</td> <th>  Prob(JB):          </th> <td>   0.810</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.969</td> <th>  Cond. No.          </th> <td>    1.01</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



When using principal component regression, in this case with only 3 components, we get an $R^2$ of $0$.

Why is that?

In constructing the problem, we scaled the uninformative components such that they have a standard deviation of $10$ and the informative component has a standard deviation of $0.1$.
PCA now focuses on the variance and leaves out the informative component.

As we can see, this is a huge problem, as we are not able to capture the important information in our data.
Now with non-standardized features, this is kind of obvious.

> But what if we standardize our features?


```python
standardized_X = StandardScaler().fit_transform(X)

stand_pc_X = PCA(n_components=3).fit_transform(standardized_X)

OLS(y, stand_pc_X).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.446</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.446</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   2682.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 07 Feb 2024</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:53:24</td>     <th>  Log-Likelihood:    </th>          <td> -11219.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 10000</td>      <th>  AIC:               </th>          <td>2.244e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  9997</td>      <th>  BIC:               </th>          <td>2.247e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>   -0.3691</td> <td>    0.007</td> <td>  -50.716</td> <td> 0.000</td> <td>   -0.383</td> <td>   -0.355</td>
</tr>
<tr>
  <th>x2</th> <td>   -0.4361</td> <td>    0.007</td> <td>  -59.713</td> <td> 0.000</td> <td>   -0.450</td> <td>   -0.422</td>
</tr>
<tr>
  <th>x3</th> <td>    0.3199</td> <td>    0.007</td> <td>   43.673</td> <td> 0.000</td> <td>    0.306</td> <td>    0.334</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.333</td> <th>  Durbin-Watson:     </th> <td>   2.005</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.513</td> <th>  Jarque-Bera (JB):  </th> <td>   1.363</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.025</td> <th>  Prob(JB):          </th> <td>   0.506</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.972</td> <th>  Cond. No.          </th> <td>    1.01</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Running the PCA *after standardizing the inputs* (which is of course important!), improves the regression.
Now, all the features, informative or not, have the same standard deviation ($=1$) and mean ($=0$).

However, we still only get an $R^2$ of $\approx 0.44$.
Remember, simple OLS gave us an $R^2$ of $1$.

This is because each of the principal components captures a part of the informative feature (which is actually equal to the target) but also a lot of noise.
The linear regression tries its best to filter out the noise but, as we can see, only manages to do so to a certain extent.

So we can see that in this example, the PCA *mixed our informative feature with noise* and therefore *made it harder for the model* to capture the important information.


## How can we avoid this problem?

1. First run a normal regression, to get a sense of the importance of your features.
2. Remove features that are not important (and noisy), standardize the features that are left.
3. Run a regression with the remaining features.

If you are experiencing problems with overfitting, I recommend using Lasso or Ridge regression.
But this should be done *after you found your (somewhat) informative features*.

Model fitting is always the last stage of a data science project, not the first.
And good features always outperform any regularization method!

If you liked this guide, please share it with your friends and colleagues.
In that sense, happy feature selection!
