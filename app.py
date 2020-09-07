from flask import Flask, request, render_template, url_for
from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, string

import nltk
import numpy as np
import random

#import pandas as pd

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

def searcharticle(query):
    list1 = []
    # opening the text file 
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\nutrition.txt','r') as file: 
       
        # reading each line     
        for line in file: 
       
            # reading each word         
            for word in line.split(): 
       
                # displaying the words            
                list1.append(word)
    print(len(list1))
    list2 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\education.txt','r') as file:    
        for line in file: 
            for word in line.split(): 
                list2.append(word)
    print(len(list2))
    
    list3 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\economy.txt','r') as file:    
        for line in file: 
            for word in line.split(): 
                list3.append(word)
    print(len(list3))
    
    list4 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\entertainment.txt','r') as file:    
        for line in file: 
            for word in line.split(): 
                list4.append(word)
    print(len(list4))
    
    list5 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\fitness.txt','r') as file:    
        for line in file: 
            for word in line.split(): 
                list5.append(word)
    print(len(list5))
    
    list6 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\origin.txt') as file:    
        for line in file: 
            for word in line.split(): 
                list6.append(word)
    print(len(list6))
    
    list7 = []
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\symptoms.txt') as file:    
        for line in file: 
            for word in line.split(): 
                list7.append(word)
    print(len(list7))
        
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 
    x = stopwords.words('english')
    x.extend(['covid-19', 'covid19' ,'corona', 'coronavirus', 'corona virus', 'covid 19', 'covid virus', 'covid'])
    
    stop_words = set(x)
    filterl1 = [w for w in list1 if not w in stop_words] 
    filterl2 = [w for w in list2 if not w in stop_words] 
    filterl3 = [w for w in list3 if not w in stop_words] 
    filterl4 = [w for w in list4 if not w in stop_words] 
    filterl5 = [w for w in list5 if not w in stop_words] 
    filterl6 = [w for w in list6 if not w in stop_words] 
    filterl7 = [w for w in list7 if not w in stop_words] 
    
    query = query.lower()
    query2=""
    #remove punctuations
    #query2 = "".join(u for u in query if u not in ["?", ".", ";", ":", "!", "/", "*", ",","[","]","(", ")","{","}","+","-","="])
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    query2 = "".join(u for u in query if u not in punctuations)
    word_tokens = word_tokenize(query2)
    filterq = [w for w in word_tokens if w not in stop_words] 
    
    count1= []
    count2= []
    count3= []
    count4= []
    count5= []
    count6= []
    count7= []
    for x in filterq:
        count1.append(filterl1.count(x))
        count2.append(filterl2.count(x))
        count3.append(filterl3.count(x))
        count4.append(filterl4.count(x))
        count5.append(filterl5.count(x))
        count6.append(filterl6.count(x))
        count7.append(filterl7.count(x))
    counts = [count1, count2, count3, count4, count5, count6, count7]
    
    result=[]
    for i in range(0,7):
        if sum(counts[i])>0:
            result.append("true")
        else:
            result.append("false")
    
    return result

@app.route('/')
def index():
    #return "Hello world!"
    posreviews=[]
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyPositive.txt','r') as file:    
        for line in file: 
            posreviews.append(line)
    poscount = len(posreviews)
    
    negreviews=[]
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyNegative.txt','r') as file:    
        for line in file: 
            negreviews.append(line)
    negcount = len(negreviews)

    return render_template('index.html', pos=poscount, neg=negcount)


@app.route('/fetchpositive', methods=['GET', 'POST'])
def fetchpositive():
    file = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyPositive.txt','r')
    postext=file.read()
    postext = postext.replace('\n',"<br>")
    return render_template('index.html', positivetext=postext)

@app.route('/fetchnegative', methods=['GET', 'POST'])
def fetchnegative():
    file = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyNegative.txt','r')
    negtext=file.read()
    negtext = negtext.replace('\n',"<br>")

    return render_template('index.html', negativetext=negtext)




@app.route('/predict',methods=['POST'])
def predict():
    
    review = request.form['review']
    
    res=""
    # load the dataset
    labels, texts = [], []
    
    datapos = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\positive.txt').read()
    for i, line in enumerate(datapos.split("\n")):
        labels.append(1)
        texts.append(line)
        
    dataneg = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\negative.txt').read()
    for i, line in enumerate(dataneg.split("\n")):
        labels.append(0)
        texts.append(line)
    
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = train_test_split(trainDF['text'], trainDF['label'])
    
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    
    #model building
    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        
        accuracy = metrics.accuracy_score(predictions, valid_y)
        print(accuracy)
        return classifier
    
    # Naive Bayes on Word Level TF IDF Vectors
    print ("NB, WordLevel TF-IDF: ")
    model1 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    
    def predict(classifier, sent, validx):
        validx[len(validx)]=sent
        xvalidTfidf =  tfidf_vect.transform(validx)
        predictions = classifier.predict(xvalidTfidf)
    
        return predictions
        
    input_sent = review
    pred_result1 = predict(model1,input_sent, valid_x)
    #pred_result2 = predict(model2, input_sent)
    if pred_result1[-1] == 1:
        res="Positive"
    else:
        res="Negative"
    
    thanks = "Thank You for Your Response!"
    show=1
    
    if res=="Positive":
        
        #taking count of positive reviews
        file = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyPositive.txt','r')
        x=file.read()
        
        x=x+"\n"
        x=x+review
        
        with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyPositive.txt','w') as filepos:    
            filepos.write(x)
        
    posreviews=[]
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyPositive.txt','r') as file:    
        for line in file: 
            posreviews.append(line)
    poscount = len(posreviews)

    if res=="Negative":
        #taking count of negative reviews
        file = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyNegative.txt','r')
        x=file.read()
        
        x=x+"\n"
        x=x+review
        
        with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyNegative.txt','w') as filepos:    
            filepos.write(x)
        
    negreviews=[]
    with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\alreadyNegative.txt','r') as file:    
        for line in file: 
            negreviews.append(line)
    negcount = len(negreviews)


    return render_template('index.html', prediction_text=res, thankyou_text=thanks, showReview=show, pos = poscount, neg=negcount)


@app.route('/search',methods=['POST'])
def search():
    
    query = request.form['query']
    result = searcharticle(query)
    
    return render_template('index.html', nutrition = result[0], education = result[1], economy=result[2], entertainment=result[3], fitness=result[4], origin=result[5], symptoms=result[6] )



@app.route('/predict1',methods=['POST'])
def predict1():
    
    df1 = pandas.read_csv(r"C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\community.csv")
    df2 = pandas.read_csv(r"C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\news.csv")
    
    Questions = []
    
    df1['title']=df1['title'].apply(str)
    for i in range(0, len(df1['title'])):
        Questions.append(df1['title'][i].lower())
    
    df2['question']=df2['question'].apply(str)
    for i in range(0, len(df2['question'])):
        Questions.append(df2['question'][i].lower())
    
    Answers = []
    
    df1['answer']=df1['answer'].apply(str)
    for i in range(0, len(df1['answer'])):
        #ans = re.sub("[^a-zA-Z]","x",str(location))
        Answers.append(df1['answer'][i].lower())
        
    df2['answer']=df2['answer'].apply(str)
    for i in range(0, len(df2['answer'])):
        Answers.append(df2['answer'][i].lower())
        
    
    nltk.download('punkt') # first-time use only
    nltk.download('wordnet') # first-time use only
    ques_tokens=[]
    
    for i in range(0, len(Questions)):
        word_tokens = nltk.word_tokenize(Questions[i])# converts each question to words tokens
        ques_tokens.append(word_tokens)
    
    
    lemmer = nltk.stem.WordNetLemmatizer()
    #WordNet is a semantically-oriented dictionary of English included in NLTK.
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
    def response(user_response):
        robo_response=''
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        
        tfidf_user = TfidfVec.fit_transform([user_response])
        cos_value = -1
        ques_idx = 0;
        for i in range(0, len(Questions)):
            tfidf_i = TfidfVec.fit_transform([Questions[i]])
            cos_new_value = cosine_similarity(tfidf_i, tfidf_user)
            if cos_new_value>cos_value:
                cos_value = cos_new_value
                ques_idx=i
    
        robo_response = Answers[i]
        matched_question = Questions[i]
        return (robo_response, matched_question)
    
    def response(user_response):
        robo_response=''
        Questions.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(Questions)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        #flat = vals.flatten()
        #flat.sort()
        #req_tfidf = flat[-2]
        Questions.remove(user_response)
        #return vals
    
        #if(req_tfidf==0):
        #    robo_response=robo_response+"I am sorry! I don't understand you"
        #    return robo_response
        #else:
        if vals[0][idx]<0.2:
            robo_response = robo_response+"Sorry we found no appropriate answers for your query. However you can check the articles below!"
        else:
            robo_response = robo_response+Answers[idx]
    
        print(vals[0][idx])
        print("matched question is:")
        print(Questions[idx])
        return robo_response
    
    ques = request.form['ques']
    result = response(ques)
    
    result2 = searcharticle(ques)
    
    return render_template('index.html', chatbot_ans=result, nutrition1 = result2[0], education1 = result2[1], economy1=result2[2], entertainment1=result2[3], fitness1=result2[4], origin1=result2[5], symptoms1=result2[6] )





@app.route('/nutrition', methods=['GET', 'POST'])
def nutrition():
    return render_template('nutrition.html')

@app.route('/education', methods=['GET', 'POST'])
def education():
    return render_template('education.html')

@app.route('/economy', methods=['GET', 'POST'])
def economy():
    return render_template('economy.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    return render_template('symptoms.html')

@app.route('/origin', methods=['GET', 'POST'])
def origin():
    return render_template('origin.html')

@app.route('/entertainment', methods=['GET', 'POST'])
def entertainment():
    return render_template('entertainment.html')

@app.route('/fitness', methods=['GET', 'POST'])
def fitness():
    return render_template('fitness.html')    

def find_summary(x):
    if x==1:
        with open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\Datasets\nutrition.txt', 'r') as file:
            article_text = file.read().replace('\n', '')
    
    import nltk
    nltk.download('punkt') # one time execution
    import re
    
    article_text_2 = ""

    from nltk.tokenize import sent_tokenize
    sentences = []
    
    sentences.append(sent_tokenize(article_text))
    
    sentences = [y for x in sentences for y in x] # flatten list
    
    len(sentences)
    
    # make alphabets lowercase
    clean_sentences = pandas.Series(sentences).str.replace("WHO", "World Health Organisation")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = pandas.Series(sentences).str.replace("covid-19", "covid")
    clean_sentences = pandas.Series(sentences).str.replace("covid19", "covid")
    
    # remove punctuations, numbers and special characters
    clean_sentences = pandas.Series(sentences).str.replace("[^a-zA-Z0-9]", " ")
    clean_sentences = pandas.Series(clean_sentences).str.replace("0","zero")
    clean_sentences = pandas.Series(clean_sentences).str.replace("1","one")
    clean_sentences = pandas.Series(clean_sentences).str.replace("2","two")
    clean_sentences = pandas.Series(clean_sentences).str.replace("3","three")
    clean_sentences = pandas.Series(clean_sentences).str.replace("4","four")
    clean_sentences = pandas.Series(clean_sentences).str.replace("5","five")
    clean_sentences = pandas.Series(clean_sentences).str.replace("6","six")
    clean_sentences = pandas.Series(clean_sentences).str.replace("7","seven")
    clean_sentences = pandas.Series(clean_sentences).str.replace("8","eight")
    clean_sentences = pandas.Series(clean_sentences).str.replace("9","nine")
    
    len(clean_sentences)
    
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    
    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    
    f = open(r'C:\Users\hp\Desktop\FALL-2020-21\Natural Language Processing-CSE4022\Project\Covid19-nlp\glove.6B\glove.6B.100d.txt')
    word_embeddings = {}
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)
      
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    
    #!pip install networkx
    
    #Applying the page rank algorithm
    import networkx as nx
    import numpy
    nx_graph = nx.from_numpy_matrix(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    #Summary Extraction
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    # Extract top 8 sentences as the summary
    ans = []
    for i in range(10):
      ans.append(ranked_sentences[i][1])
      
    ansstr = ""
    for line in ans:
        ansstr = ansstr+line
        ansstr = ansstr+"\n"
    
    return ansstr
        
        
   
@app.route('/summarise1', methods=['GET', 'POST'])
def summarise1():
    #answer = find_summary(1)
    answer = 'keep your hands, kitchen and utensils clean, separate raw and cooked food, especially raw meat and fresh produce, cook your food thoroughly, keep your food at safe temperatures, either below 5 \xc2\xb0C or above 60 \xc2\xb0C; and use safe water and raw material. \n\nThe availability of fresh foods may decrease and it may therefore become necessary to rely more on canned, frozen or processed foods.\n\n In many countries, 50\xe2\x80\x9375% of the salt intake comes from the foods we eat, rather than what we add ourselves.\n\n This way you can avoid food waste and allow others to access the food they need.Use fresh ingredients and those that have a shorter shelf life first.\nFrozen fruits and vegetables can also conveniently be used over longer periods of time and often have a similar nutrient profile to fresh foods.\n\n You might feel the need to purchase large amounts of foods, but make sure to consider and utilize what is already in your pantry, as well as foods with shorter shelf life.\n\n Experiment with fresh or dried herbs and spices for added flavour instead.WHO recommends that ideally less than 5% of total energy intake for adults should come from free sugars (about 6 teaspoons).\n\n Some examples of healthy recipes with accessible ingredients may also be found below.It can be difficult to get portion sizes right, especially when cooking from scratch.\n\n When other dessert options are chosen, ensure that they are low in sugar and consume small portions.\n\n To avoid food waste, you may consider freezing any leftovers for another meal.During regular daily life, many individuals often do not have the time to prepare home-cooked meals.\n\n If fresh products, especially fruits, vegetables and reduced-fat dairy products continue to be available, prioritize these over non-perishables.\n\n'
    heading = 'Recommended Nutrition during Covid-19'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise2', methods=['GET', 'POST'])
def summarise2():
    #answer = find_summary(2)
    answer = 'This has placed extra stress on families and may even become a financial burden to some.Although students have lost the one-on-one learning experience offered in the classroom, several online learning platforms are seeking to help students finish the semester strong by offering free online courses.Worldwide learning platforms, including Coursera and Khan Academy, are providing everything from online college courses to homework problem assistance.\n\nAlthough the educational system was unequipped to move 100 percent online, perhaps this experience will force schools to invest in an easily transferable digital platform for coursework in the future.It\xe2\x80\x99s no secret that most parents aren\xe2\x80\x99t exactly delighted to fill the role of teacher this semester \xe2\x80\x93 especially those who are trying to work from home.\n\nDuring this time, teachers and administrators were understandably ill prepared as many scrambled to establish an online platform that could deliver the same quality of education as before.This delay in courses has pushed the majority of student course schedules deeper into the summer.\n\nHowever, given the situation and lack of alternative options, schools around the world have begun offering more online courses than ever before.In fact, there\xe2\x80\x99s not only more online college courses available but some schools are providing discounts as well.Spring and summer time are usually when universities host a number of activities on their campuses such as camps, meetings, and other ancillary activities.\n\nWebsites such as CampusTours are offering virtual tours of more than 1,800 schools in the United States, as well as tours of schools in the United Kingdom, Canada, China, and France.Other resourceful platforms include StriveScan, which is offering students a chance to ask questions to officials from more than 450 colleges from 45 states and 13 countries on everything from college essay advice to applying for financial aid.Before the spread of COVID-19, many colleges and universities had limited online course options, especially over the summer.\n\n'
    heading = 'Impact of Covid-19 on Education'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise3', methods=['GET', 'POST'])
def summarise3():
    #answer = find_summary(3)
    answer = 'Around 65 to 70% of active pharmaceutical ingredients and around 90% of certain mobile phones come from China to India.Therefore, we can say that due to the current outbreak of coronavirus in China, the import dependence on China will have a significant impact on the Indian industry.In terms of export, China is India\xe2\x80\x99s 3rd largest export partner and accounts for around 5% share.\n\nIndia\xe2\x80\x99s electronic industry may face supply disruptions, production, reduction impact on product prices due to heavy dependence on electronics component supply directly or indirectly and local manufacturing.The New Year holidays in China has been extended due to coronavirus outbreak that adversely impacted the revenue and growth of Indian IT companies.Due to the coronavirus outbreak, the inflow of tourists from China and from other East Asian regions to India will lose that will impact the tourism sector and revenue.An outbreak of COVID-19 impacted the whole world and has been felt across industries.\n\nIt is said that the government should take some strong fiscal stimulus to the extent of 1% of GDP to the poor, which would help them financially and also manage consumer demand.In the third quarter (October-December) growth is slowed down to 4.7% and the impact of COVID-19 will further be seen in the fourth quarter.Ficci survey showed 53% of Indian businesses have indicated a marked impact of COVID-19 on business operations.\n\n"And there is urgent need to mobilise resources to stimulate the economy for increased demand and employment".According to the KPMG report "It is expected that the course of economic recovery in India will be smoother and faster than that of many other advanced countries".In terms of trade, China is the world\xe2\x80\x99s largest exporter and second-largest importer.\n\nAlso, we can\'t ignore that the lockdown and pandemic hit several sectors including MSME, hospitality, civil aviation, agriculture and allied sector.According to KPMG, the lockdown in India will have a sizeable impact on the economy mainly on consumption which is the biggest component of GDP.Reduction in the urban transaction can lead to a steep fall in the consumption of non-essential goods.\n\nFurther, 34 per cent said that exports would take a hit by more than 10 per cent.According to Du & Bradstreet, COVID-19 no doubt disrupted human lives and global supply chain but the pandemic is a severe demand shock which has offset the green shoots of recovery of the Indian economy that was visible towards the end of 2019 and early 2020.\n\nOverall, the impact of coronavirus in the industry is moderate.According to CLSA report, pharma, chemicals, and electronics businesses may face supply-chain issues and prices will go up by 10 percent.\n\nSectors that would be much affected includes logistics, auto, tourism, metals, drugs, pharmaceuticals, electronic goods, MSMEs and retail among others.Further, according to the World Bank\'s assessment, India is expected to grow 1.5 per cent to 2.8 per cent.\n\nFurther, 70 per cent of the surveyed firms are expecting a degrowth sales in the fiscal year 2020-21.Ficci said in a statement, "The survey clearly highlights that unless a substantive economic package is announced by the government immediately, we could see a permanent impairment of a large section of the industry, which may lose the opportunity to come back to life again.\n\nSome commodities like metals, upstream and downstream oil companies, could witness the impact of lower global demand impacting commodity prices.According to CII, GDP could fall below 5% in FY 2021 if policy action is not taken urgently.\n\n'
    heading = 'Impact of Covid-19 on Indian Economy'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise4', methods=['GET', 'POST'])
def summarise4():
    #answer = find_summary(4)
    answer = 'According to a study published in 2019, Angiotensin converting enzyme 2 (ACE.2), a membrane exo-peptidase in the receptor used by corona virus in entry to human cells.According to a report published on 24 Jan 2020, corona virus infected patient have many common features such as fever, cough, and fatigue while diarrhea and dyspnea were found to be as uncommon feature.\n\nAnother study reported about airborne transmission of virus while no one was presents the solid evidence.\n\nStill health professionals were not fully satisfied with any therapy so further clinical research needed.Corona virus was spreading human to human to transmission by close contact via airborne droplets generating by coughing, sneezing, kissing and smooching.\n\nAc-cording to the Canadian study 2001, approximately 500 patients were identified as Flu-like system.\n\nAfter a deep exercise they conclude and understand the patho-genesis of disease and discovered as corona virus.\n\nAccording to WHO, some general guidelines were published such as separate the infected patient from other family member to single room, implementation of contact and droplet precaution, airborne precaution etc.\n\nAvoid visiting markets and places where live or dead animals are handled, Wash your hands with soap and water or use an alcohol based disin-fectant solution before eating, after using the toilet and af-ter any contact with animals, Avoid contact with animals, their excretions or droppings.There is no special vaccine for this yet.\n\nIn 2012, Saudi Arabian reports were presented several infected patient and deaths.\n\nTill now, corona virus was not confirmed in feaces and urine sample of patent.There is nothing to provide complete guidance to prevent from corona virus but some guidelines was presented by WHO and ECDC.\n\nThere are no anti corona virus vaccine to prevent or treatment but some supporting therapy work.\n\nCorona was treated as simple non fatal virus till 2002.\n\n'
    heading = 'History and Origin of Corona Virus'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise5', methods=['GET', 'POST'])
def summarise5():
    #answer = find_summary(5)
    answer = 'If you are unfortunately having to spend time alone at home, then there is always patience.Monopoly \xe2\x80\x93 careful not to let things become too heated, a board game that perhaps should come with a health warning.\n\nWe all deserve a treat or two in the current uncertain times, so now could be the time to find your inner Bake Off skills.Getting lost in a good book need not be a solitary experience.\n\nHopefully you have enough to spot out of your windows to make it a bit more interesting.Up the drama in your home by taking on the challenge of staging your own play.\n\nFailing that, you can go for an a cappella recital or find one of several karaoke videos on YouTube.Granted, you may run out of things to spot after a while, but I spy is another classic game for everyone to take part in.\n\nReading a bedtime story to children or sharing a tale with the wider family can be a calming way to spend time together.Music has the power to be uplifting, and singing or playing together may help raise spirits under lockdown.\n\nYou could also try out consequences, where you take it in turns to contribute sentences to a story, or categories, which involves thinking up things beginning with a certain letter under a variety of subjects.There are hundreds, if not thousands, of variants of card games out there.\n\nSnapping up properties, making money and getting out of jail can be great fun, but may also lead to a little family tension at times.\n\n'
    heading = 'Recreation and Entertainment during Lockdown'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise6', methods=['GET', 'POST'])
def summarise6():
    #answer = find_summary(6)
    answer = 'Even though laziness seems the new normal amidst others \xe2\x80\x98newness\xe2\x80\x99 that staying at home has brought, there\xe2\x80\x99s no substitute to self-care and fitness in every way \xe2\x80\x93 physical, mental and emotional.Physical exercise has obvious benefits for the body.\n\nYou can add a mix of push-ups (regular and advanced), circuit training (a series of exercises in rapid succession with little or no breaks in between) and more.Push-ups may appear deceptively simple but they can really pack a punch and help you sculpt your body, particularly your upper body and core.\n\nWhile there are go-getters who always make the most of the situation at hand, and more so during quarantine when the world is finding ways to stay motivated through social media, writing, reading, binge-watching and probably even working on one\xe2\x80\x99s sleep cycle, an important aspect of good health once again.\n\nThis is the perfect way to calm a restless mind and reconnect with yourself.You can also try Guided meditation through content available on YouTube (including Sadhguru\xe2\x80\x99s guide via The Isha Foundation), Spotify and meditation apps such as Headspace or Calm to name a few.The best part of it all: you\xe2\x80\x99re at home and you\xe2\x80\x99re working on staying healthy, which is the first part toward battling any health-related ailments and lifestyle disorders.\n\nThat rush of endorphin and dopamine when you finish a challenging yet satisfying workout are just what you need to stay mentally and emotionally fit.Three basic types of home workout are: Using your bodyweight alone; Using basic equipment such as Dumbbells, Resistance Bands/Tubes; Using everyday household items such as backpacks, water bottles, buckets and even brooms.It\xe2\x80\x99s a good idea to start with basic bodyweight exercises and gradually add progressions and increase the intensity.\n\nThis way, the difficulty levels are increased slowly which helps you break a sweat and strengthen your muscles while also avoiding injuries.\n\n'
    heading = 'Fitness and Exercise during Lockdown'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise7', methods=['GET', 'POST'])
def summarise7():
    #answer = find_summary(7)
    answer = ''
    heading = 'Symptoms and Precaution of Covid-19'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)

@app.route('/summarise8', methods=['GET', 'POST'])
def summarise8():
    #answer = find_summary(8)
    answer = ''
    heading = 'Lockdown Rules in India'
    return render_template('summaryresult.html', summary_result = answer, heading = heading)






if __name__ == "__main__":
    app.run(debug=True)
    