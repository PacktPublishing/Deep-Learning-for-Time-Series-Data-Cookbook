# Deep Learning For Time Series Cookbook


<a href="https://www.packtpub.com/product/deep-learning-for-time-series-cookbook/9781805129233"><img src="https://m.media-amazon.com/images/I/81fvz3hf2rL._SL1500_.jpg" alt="Deep Learning For Time Series Cookbook" height="256px" align="right"></a>

This is the code repository for [Deep Learning For Time Series Cookbook](https://www.packtpub.com/product/deep-learning-for-time-series-cookbook/9781805129233), published by Packt.

**Use PyTorch and Python recipes for forecasting, classification, and anomaly detection**

## What is this book about?

Many organizations, for example in finance, have some element of time dependency in their structures and processes. By leveraging time series analysis and forecasting, these organizations can make informed decisions and optimize their performance. Accurate forecasts help reduce uncertainty and enable better planning of operations. Unlike traditional approaches to forecasting, deep learning can process large amounts of data and help derive complex patterns. Despite its increasing relevance, getting the most out of deep learning requires significant technical expertise.
 
This book covers the following exciting features: 
* Grasp the core of time series analysis and unleash its power using Python
* Understand PyTorch and how to use it to build deep learning models
* Discover how to transform a time series for training transformers
* Understand how to deal with various time series characteristics
* Tackle forecasting problems, involving univariate or multivariate data
* Master time series classification with residual and convolutional neural networks
* Get up to speed with solving time series anomaly detection problems using autoencoders and generative adversarial networks (GANs)

If you feel this book is for you, get your [copy](https://www.amazon.com/Deep-Learning-Time-Data-Cookbook/dp/1805129236/ref=sr_1_1?sr=8-1) today!

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(x=series_daily, model='additive', period=365)
```

**Following is what you need for this book:**
If you’re a machine learning enthusiast or someone who wants to learn more about building forecasting applications using deep learning, this book is for you. Basic knowledge of Python programming and machine learning is required to get the most out of this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
|  	1-9   |    Python (3.9)                             			  | Windows, macOS, or Linux | 		
|  	1-9   |   PyTorch Lightning (2.1.2)                              			  | Windows, macOS, or Linux | 		
|  	1-9   |   pandas (>=2.1)                              			  | Windows, macOS, or Linux | 		
|  	1-9   |   scikit-learn (1.3.2)                              			  | Windows, macOS, or Linux | 		
|  	1-9   |  NumPy (1.26.2)                               			  | Windows, macOS, or Linux | 		
|  	1-9   |  torch (2.1.1)                               			  | Windows, macOS, or Linux | 		
|  	1-9   |  PyTorch Forecasting (1.0.0)                               			  | Windows, macOS, or Linux | 		
|  	1-9   |  GluonTS (0.14.2)                               			  | Windows, macOS, or Linux | 		


### Related products <Other books you may enjoy>
* Time Series Analysis with Python Cookbook  [[Packt]](https://www.packtpub.com/product/time-series-analysis-with-python-cookbook/9781801075541) [[Amazon]](https://www.amazon.com/Time-Analysis-Python-Cookbook-exploratory/dp/1801075549/ref=sr_1_1?sr=8-1)
  
* Modern Time Series Forecasting with Python  [[Packt]](https://www.packtpub.com/product/modern-time-series-forecasting-with-python/9781803246802) [[Amazon]](https://www.amazon.com/Modern-Time-Forecasting-Python-industry-ready/dp/1803246804/ref=sr_1_1?sr=8-1)
  
## Get to Know the Author
**Vitor Cerqueira** is a machine learning researcher at the Faculty of Engineering of the University of Porto, working on a variety of projects concerning time series data, including forecasting, anomaly detection, and meta-learning. Vitor has earned his Ph.D. with honors from the University of Porto in 2019, and also has a background on data analytics and mathematics. He has authored several peer-reviewed publications on related topics.

**Luís Roque**, is the Founder and Partner of ZAAI, a company focused on AI product development, consultancy, and investment in AI startups. He also serves as the Vice President of Data & AI at Marley Spoon, leading teams across data science, data analytics, data product, data engineering, machine learning operations, and platforms. In addition, he holds the position of AI Advisor at CableLabs, where he contributes to integrating the broadband industry with AI technologies.
Luís is also a Ph.D. Researcher in AI at the University of Porto's AI&CS lab and oversees the Data
Science Master's program at Nuclio Digital School in Barcelona. Previously, he co-founded HUUB,
where he served as CEO until its acquisition by Maersk.
