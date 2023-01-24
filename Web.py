import main
from flask import Flask, render_template, redirect, url_for, request,jsonify
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
app = Flask(__name__)
app_name ="Finance Sentiment Analysis"
@app.route("/")
def hello_world():
    return render_template('index.html',app_name=app_name)

@app.route('/_predict_search')
def pred():
    query = request.args.get('a', 0, type=str)
    
    key = ['finance' , 'Finances' , 'Finance' , 'FINANCE' , 'FINANCES' , 'finances' , 'companies' , 'Companies' , 'company', 'Company' , 'financial' , 'Financial' , 'Financials' , 'financials' , 'google' , 'Google' , 'yahoo' , 'Yahoo' , 'share' , 'Share' , 'markets' , 'Markets' , 'market' , 'Market' , 'news' , 'News' , 'NEWS' , 'efinance' , 'exeter' , 'prices' , 'price' , 'Prices' , 'Price' , 'ministry' , 'Ministry' , 'quotes' , 'finanzen' , 'financas' , 'toyota' , 'Toyota' , 'planning' , 'Planning' , 'institutions' , 'Institutions' , 'institution' , 'Institutions' , 'planner' , 'Planner' , 'wealth' , 'Wealth' , 'management' , 'Management' , 'department' , 'Department' , 'advisor' , 'Advisor' , 'cfp' , 'CFP' , 'services' , 'Services' , 'service' , 'Service' , 'honda' , 'Honda' , 'ifa' , 'IFA' , 'investment' , 'Investment' , 'mutual funds' , 'Mutual funds' , 'Mutual Funds' , 'insurance' , 'Insurance' , 'fsa' , 'FSA', 'mortgage' , 'Mortgage' , 'mortgages' , 'Mortgage' , 'loan', 'Loan', 'Loans' , 'loans' , 'investing' , 'Investing' , 'invests' , 'Invest' , 'crash' , 'crashes' , 'crypto' , 'regulations' , 'regulation' , 'audit' , 'director' , 'former' , 'deputy' , 'resign' , 'officials' , 'rise' , 'supports' , 'exchange' , 'executives' , 'dollars' , ' Crypto ' , ' Regulations ' , ' Audit ' , ' Regulation' , ' Director ' , ' Former ' , ' Deputy ' , ' Resign ' , ' Officials ' , ' official' , ' Rise ' , ' rises ' , ' Rises ' , ' Supports ' , ' support ' , ' Support ' , ' Exchange ' , ' Executives ' , ' Executive' , ' executive' , ' Dollars ' , 'Client' , 'client' , 'clients' , 'Clients' , 'Compilance' , ' compilance ' , ' Compilances ' ,' compilances ' , 'equity' , 'excel' , 'licenses' , 'models' , 'performance' , 'portfolio' , ' Equity ' , ' Excel ' , ' Licenses ' , ' license ' , ' Licenses ' , ' Models ' , ' model ' , ' Model ' , ' Performance ' , ' performances ' , ' Performances ' , ' Portfolio ' , 'research' ,'review' , 'valuation' , ' Research ' , ' corporate ' , ' Corporate ' , 'corporates' , 'Corporates' , 'public' ,'Public' ,'Accomplished', 'Achieved','Adapted','Advanced','Advised','Analyzed','Arbitrated','Assessed','Assured','Attained','Balanced','Brainstormed','Budgeted','Built','Calculated','Centralized','Championed','Changed','Clarified','Coached','Collaborated','Communicated','Completed','Complied','Conceived','Conducted','Constructed','Consulted','Converted','Coordinated','Created','Decreased','Delegated','Delivered','Designed','Determined','Developed','Devised','Directed','Disbursed','Distributed','Documented','Earned','Edited','Eliminated','Engaged','Energized','Enhanced','Established','Examined','Exceeded','Excelled','Executed','Expedited','Extracted','Evaluated','Facilitated','Finalized','through','Forecasted','Formed','Fulfilled','Gained','Generated','Handled','Headed','Identified','Implemented','Improved','Increased','Initiated','Innovated','Installed','Instituted','Integrated','Introduced','Investigated','Launched','Lead','Leverage','Liquidated','Maintained','Managed','Maximized','Mentored','Merged','Monitored','Negotiated','Operated','Optimized','Organized','Overcame','Oversaw','Partnered','Performed','Piloted','Pinpointed','Planned','Positioned','Predicted','Prepared','Presented','Presided','Prevented','Prioritized','Processed','Produced','Projected','Promoted','Qualified','Quantified','Recommended','Reconciled','Rectified','Redefined','Redesigned','Reduced','Regulated','Related','Reorganized','Researched','Resolved','Restored','Restructured','Revamped','Revitalized','Salvaged','Satisfied','Saved','Sold','Solidified','Spearheaded','Strategized','Streamlined','Structured','Supervised','Tailored','Taught','Tracked','Traded','Trained','Transferred','Uncovered','Unified','Upgraded','Utilized','Validated','Verified','Wrote','accomplished','achieved','adapted','advanced','advised','analyzed','arbitrated','assessed','assured','attained','balanced','brainstormed','budgeted','built','calculated','centralized','championed','changed','clarified','coached','collaborated','communicated','completed','complied','conceived','conducted','constructed','consulted','converted','coordinated','created','decreased','delegated','delivered','designed','determined','developed','devised','directed','disbursed','distributed','documented','earned','edited','eliminated','engaged','energized','enhanced','established','examined','exceeded','excelled','executed','expedited','extracted','evaluated','facilitated','finalized','through','forecasted','formed','fulfilled','gained','generated','handled','headed','identified','implemented','improved','increased','initiated','innovated','installed','instituted','integrated','introduced','investigated','launched','lead','leverage','liquidated','maintained','managed','maximized','mentored','merged','monitored','negotiated','operated','optimized','organized','overcame','oversaw','partnered','performed','piloted','pinpointed','planned','positioned','predicted','prepared','presented','presided','prevented','prioritized','processed','produced','projected','promoted','qualified','quantified','recommended','reconciled','rectified','redefined','redesigned','reduced','regulated','related','reorganized','researched','resolved','restored','restructured','revamped','revitalized','salvaged','satisfied','saved','sold','solidified','spearheaded','strategized','streamlined','structured','supervised','tailored','taught','tracked','traded','trained','transferred','uncovered','unified','upgraded','utilized','validated','verified','wrote']
    res = any(item in query for item in key) 
    if res == True:
        Answer = main.get_value(str_var=query).split('\n')
        
    else:
        Answer = ["Not Valid Statement","","",""]
     
    print(Answer)   
    return jsonify(result=Answer)


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        print("Error Occured!")
    
