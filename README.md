# Predicting Mortgage Foreclosures

<h3>Problem Statement</h3>
<p>This project focuses on developing a mortgage loan default prediction model based on the Fannie Mae dataset. When a potential borrower seeks a new mortgage, the model will either categorize the loan as a likely default or non-default based on several variables to be determined by the model.</p>

<h3>Data Source</h3>
<p>Fannie Mae is a government sponsored entity that was created in 1938 to assist in the housing market recovery from the effects of the Great Depression. Fannie Mae along with a similar entity, Freddie Mac, purchases loans originated and secured through other lending entities and sells them in the bond market.
  
Fannie Mae has made available a large [dataset](https://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) of over 35 million mortgage loans however there are quite a few exclusions from this dataset including riskier loans such as interest only and balloon amortization. A complete list of exclusions can be found on the website's [FAQ](https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_FAQs.pdf).

The data files are split between two types of text files, acquisition and performance data with a text file for each quarter of the year. The acquisition files have 25 columns and the performance files have 31 columns.
</p>

<h3>Methodology</h3>

Step 1: Data Wrangling - Imported .txt files and merge into one DataFrame, converted datatypes as required, and filled missing values/dropped rows.

Step 2: Exploratory Data Analysis - Visualized data with charts and graphs to uncover insights about the data.

Step 3: Inferential Statistics - Used random sampling to test the null hypothesis that there is not a difference between credit scores of borrowers who defaulted and those who did not default.

Step 4: Machine Learning - Created a model to predict mortgage applicants whose loans will default.

