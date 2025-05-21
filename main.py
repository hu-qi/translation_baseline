# coding:utf-8
import time
import pandas as pd
from tqdm import trange
from nllb import ml_translator,th_translator,en_translator
from asr import inference_pipeline

data=pd.read_csv('testa.csv')

#语音转文本
asrs=[]
for i in trange(len(data)):
    wav = data['音频路径'][i]
    asr_result = inference_pipeline(wav)
    asrs.append(asr_result)
data['语音识别结果']=asrs

#翻译
results=[]
for i in trange(len(data)):
    text=data['语音识别结果'][i]
    language=data['语言'][i]
    if language=='泰语':
        try:
            result = th_translator(text)
        except:
            result=''
    elif language=='英语':
        try:
            result = en_translator(text)
        except:
            result = ''
    elif language=='马来语':
        try:
            result = ml_translator(text)
        except:
            result = ''
    else:
        result=''
    results.append(result)
data['answer']=results
data.to_csv('testa_predict.csv',index=False)
