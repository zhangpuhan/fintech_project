from . import app
from .forms import FeatureForm

from flask import render_template
import requests
import pandas as pd
import json

from models.review import XgbModel

@app.route('/')
def index():
    api_key = "am36v8r5maakuqSqeTok9Zgzexw="
    header = {"Authorization" : api_key}
    url = "https://api.lendingclub.com/api/investor/v1/loans/listing"
    
    r = requests.get(url, headers=header)
    data = r.json()
    MyData = data["loans"]
    with open("current_list.txt", "w") as output:
        json.dump(MyData, output)
    df_current = pd.read_json("current_list.txt")
    df_current = df_current[["id","addrState","annualInc","grade","empTitle","loanAmount","intRate","term"]]
    df_current = df_current[df_current["term"]==36]
    df_current = df_current.reset_index(drop=True)
    df_current.drop("term", inplace=True, axis=1)
    df_current.columns = ["Loan ID","Address", "Annual Income", "Grade", "Job title", "Amount", "Rate %"]
    th_props = [
                ('font-size', '13px'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('color', '#6d6d6d'),
                ('background-color', '#f7f7f9')
                ]

    # Set CSS properties for td elements in dataframe
    td_props = [('font-size', '13px')]

    # Set table styles
    styles = [
          dict(selector="th", props=th_props),
          dict(selector="td", props=td_props)
          ]
    df_html_output = df_current.style.set_table_styles(
                                               styles
                                               ).render()

    return render_template('welcome.html', data=df_html_output)



@app.route('/result/')
def result():
    okdata = XgbModel()
    datapredict = okdata.predict()
    datapredict = pd.DataFrame(datapredict, columns=["Risk"])
    datapredict["Risk"] = datapredict["Risk"].map('{:,.2f}'.format)
    
    df_current = pd.read_json("current_list.txt")
    df_current = df_current[df_current["term"]==36]
    df_current.drop("term", inplace=True, axis=1)
    df_current = df_current.reset_index(drop=True)
    df_current["risk"] = datapredict
    df_current = df_current[["id","addrState","annualInc","grade","loanAmount","intRate", "risk"]]
    
    
    df_current.columns = ["Loan ID", "Address", "Annual Income", "Grade", "Amount", "Rate %", "Risk"]
    th_props = [
                ('font-size', '13px'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('color', '#6d6d6d'),
                ('background-color', '#f7f7f9')
                ]
    
                # Set CSS properties for td elements in dataframe
    td_props = [('font-size', '13px')]
                
                # Set table styles
    
    styles = [dict(selector="th", props=th_props), dict(selector="td", props=td_props)]
    df_html_output = df_current.style.set_table_styles(styles).render()
                
    return render_template('result.html', data=df_html_output)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/author')
def author():
    return render_template('author.html')
