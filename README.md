# McGill University - Final Project  
Topic Modelling Project for YCNG 229 - Neural Networks & Deep Learning Course

This repository is for individuals who would like to find clusters from titles that are presumed to be relatively similar to each other. The objective of this project was to cluster similar financial news headlines in order to streamline prevalent topics coming from over 280 different financial news sources. 

  * Use git repo to spawn local flask web application that clusters financial news data coming from a datafile containing financial titles.
  * To display clusters in interactive circle Bokeh plot or Table Bokeh Plot.

Prerequisites:

* python >= 3.8.5
* conda installed
* See Environment.yml file for required python packages

# Code Organization 
 * `app.py`: This file contains the main for the Flask server. It is also the entrypoint of the app. It contains 1 entrypoint that takes in /model/ or /model/topics/ app routes, as parameters.
 * `getUmapEmbeddings.py`: This file is used to generate UMAP embeddings from embeddings from distilbert outputs.
 * `hdbscanClusters.py`: This file is used to generate clusters from the UMAP embeddings - cluster function is defined in this script.
 * `main.py`: Main script used to take in parameters and output topic clustered bokeh plot local html file. 
 * `ParameterTuningClusters.py`: Script to tune hyperparameters for UMAP embeddings and hdbscan clusters - goal is to maximize silhouette score for clusters. 
 * `textPreprocessing.py`: Script used to preprocess text data - news headlines before generating embeddings
 * `TopicBokehPlot.py`: Script used that contains bokeh plot function used to generate bokeh plot with user defined parameters. 
 * `topicDataframe.py`: Script used to generate table of topics with top 5 keywords per topic in dataframe to html format. 
 * `CircleBokeh.py`: Script used to generate the circle bokeh plot.
 * `TableBokeh.py`: Script used to generate the table bokeh plot. 

# APP workflow
The app is using a flask server to process the queries. When the server receive a query on `/model` which will output a circle bokeh plot of similar titles into clusters. In addition, `/model/topics` can be used to generate the top 5 keywords per cluster in a tabular format.  
  
The following will:
1. Run the app.py file to load up flask web server and then input into browser corresponding URL: `http://127.0.0.1:8080/model` or `http://127.0.0.1:8080/model/topics`
2. The data from the ...\Data\ will then be passed to textPreprocessing function and output will be a list of title. 
3. The list of titles will then go through distilbert model and generate text embeddings. 
4. Distilbert Embeddings will then be dimension reduced through UMAP embeddings implmentation. 
5. HDBSCAN clustering will occur on 2D embbedings and Bokeh topic model plot will be made on top of the UMAP Embeddings and the HDBSCAN cluster labels. 

 
# Example of Bokeh Plot Topic Modelling Clusters
![Bokehplot](https://user-images.githubusercontent.com/42786192/124517259-88dbfb00-ddb1-11eb-8194-4b2cf3405633.png)

![tableBokeh](https://user-images.githubusercontent.com/42786192/125131992-aa025b80-e0d1-11eb-8c5d-e798cb87bcb9.png)


