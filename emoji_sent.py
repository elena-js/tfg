import pandas as pd

# dataset con valores sentimiento emojis
data = pd.read_csv("emoji.csv")

emoji = data['Emoji']
code = data['Unicode codepoint']
sent = []

for i in range(0,len(emoji)):
  pos = data.iloc[i]['Positive']
  neg = data.iloc[i]['Negative']
  neu = data.iloc[i]['Neutral']
  ppos = pos / (pos+neg+neu)
  pneg = neg / (pos+neg+neu)
  pneu = neu / (pos+neg+neu)
  sent.append(ppos - pneg)
  '''
  if pos > neg and pos > neu :
    sent.append('pos')
  elif neg > pos and neg > neu :
    sent.append('neg')
  else:
    sent.append('neu')
  '''

# HACER CSV
import pandas as pd	

dict = {'Emoji': emoji, 'Unicode codepoint': code, 'sent': sent} 
df = pd.DataFrame(dict) 
df.to_csv('sent_emojis.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()